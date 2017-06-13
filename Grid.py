import numpy as np
import collections


class Grid:

    COORDINATE_SYSTEMS = ["SPHERICAL_2D"]

    def __init__(self, coordinate_system):
        # Initialization of the Grid using the hash table.
        # The hash table self.arrays contains an information about
        # the grid graphs in d dimentions.
        # Thereafter the length of the self.arrays can be less or equal to d.
        # By defining the coordinate system we later initialize the volume elements

        self.arrays = {}
        self.arrays_diff = {}
        self.coordinate_system = coordinate_system

    def init1d(self, name, grid_min, grid_max, grid_step):
        # creates 1 dimentional grid graph with equidistant spacing
        # and adds this array to the hash table self.arrays[name]

        grid_1d = np.arange(grid_min, grid_max + grid_step, grid_step)
        self.arrays_diff[name] = grid_step
        self.arrays[name] = grid_1d

    def return_array1d(self, name):
        # with a given key returns a corrsponding 1 dimentional array
        return self.arrays[name]

    def function_prod(self, list_of_unit_vectors, list_of_functions):
        # calculates a function on a grid when function is a product of function
        # acting independently on each coordinate
        # example:  list_of_unit_vectors = [x1, x2]
        # assume that f = f1 (x1) * f2 (x2)
        # then we can use: list_of_functions = [f1, f2, ...]
        # NOTE_ys: do we need to generalize it?

        # check that the order of keys in the list_of_unit_vectors is ordered
        # the same way as in self.arrays.keys
        # otherwise throw an error
        nC = collections.Counter(list_of_unit_vectors)
        sC = collections.Counter(self.arrays.keys())
        if(nC != sC):
            print('INVALID LIST OF NAMES')
            return

        outer_mat = list_of_functions[0](self.arrays[list_of_unit_vectors[0]])

        if(len(list_of_unit_vectors) == 1):
            return outer_mat

        for ind, name in enumerate(list_of_unit_vectors[1:]):
            temp = list_of_functions[ind + 1](self.arrays[name])
            outer_mat = np.outer(outer_mat, temp)

        return outer_mat.reshape(outer_mat.size)

    def dV(self):
        # create an infinitisimal element of the volume that corresponds to the
        # given coordinate_system

        list_of_unit_vectors = list(self.arrays.keys())

        # create dk, dth and modify it
        grid_diff = self.arrays_diff[list_of_unit_vectors[0]] * np.ones(len(self.arrays[list_of_unit_vectors[0]]))
        grid_diff[0] = 0.5 * grid_diff[0]
        grid_diff[-1] = 0.5 * grid_diff[-1]

        if(len(list_of_unit_vectors) == 1):
            return grid_diff

        for ind, name in enumerate(list_of_unit_vectors[1:]):
            temp_grid_diff = self.arrays_diff[name] * np.ones(len(self.arrays[name]))
            temp_grid_diff[0] = 0.5 * temp_grid_diff[0]
            temp_grid_diff[-1] = 0.5 * temp_grid_diff[-1]
            grid_diff = np.outer(grid_diff, temp_grid_diff)

        coordinate_system = self.coordinate_system
        if coordinate_system == "SPHERICAL_2D":
            list_of_functions = [lambda k: (2 * np.pi)**(-2) * k**2, np.sin]

        output = self.function_prod(list_of_unit_vectors, list_of_functions)

        #print(len(output))
        return output * grid_diff.reshape(grid_diff.size)
        # use simps method for integration since there is no dk and dth yet

    def size(self):
        # this method returns the number of grid points
        list_of_unit_vectors = list(self.arrays.keys())
        grid_size = 1
        for unit in list_of_unit_vectors:
            grid_size = grid_size * len(self.arrays[unit])
        return grid_size

    # def construct_grid(self):
    #     # construct grid from self.arrays
    #     # comment: we can use function_prod here with list_of_functions
    #     # filled with identity functions (need to write such function)

    #     list_of_unit_vectors = list(self.arrays.keys())

    #     outer_product = self.arrays[list_of_unit_vectors[0]]

    #     if(len(list_of_unit_vectors) == 1):
    #         return outer_product

    #     for ind, name in enumerate(list_of_unit_vectors[1:]):
    #         temp = self.arrays[name]
    #         outer_product = np.outer(outer_product, temp)

    #     return outer_product.reshape(outer_product.size)

    # def integrate_on_grid(self, array):
        # takes an array on a grid and integrates it
