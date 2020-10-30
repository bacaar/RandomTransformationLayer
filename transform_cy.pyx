from libc.stdint cimport intptr_t
from libc.stdint cimport uint8_t
import numpy as np
cimport numpy as np

import cython

"""
int mapping(npy_intp *output_coordinates, double *input_coordinates,
    int output_rank, int input_rank, void *user_data)
int mapping(intptr_t *output_coordinates, double *input_coordinates,
    int output_rank, int input_rank, void *user_data)
    
input coordinates are the pixels of input image
output coordinates are the pixels of output image

for each pixel of the output image we are searching the corresponding values in the input image
therefore output_coordinates are our inputs, input coordinates are our outputs
"""

cdef api int transform_func_c(intptr_t* output_coords, double* input_coords,
                              int output_rank, int input_rank, void* user_data):

    input_coords[0] = (<double*>user_data)[0] * output_coords[0] + (<double*>user_data)[1] * output_coords[1] + \
                      (<double*>user_data)[2] * output_coords[2] + (<double*>user_data)[3] * output_coords[3]

    input_coords[1] = (<double*>user_data)[4] * output_coords[0] + (<double*>user_data)[5] * output_coords[1] + \
                      (<double*>user_data)[6] * output_coords[2] + (<double*>user_data)[7] * output_coords[3]

    input_coords[2] = (<double*>user_data)[8] * output_coords[0] + (<double*>user_data)[9] * output_coords[1] + \
                      (<double*>user_data)[10] * output_coords[2] + (<double*>user_data)[11] * output_coords[3]

    return 1


@cython.boundscheck(False)
def affine_transformation_cy(np.ndarray[double, ndim=1, mode="c"] coords,
                             int n_voxel,
                             np.ndarray[double, ndim=1, mode="c"] trans_matrix):

    cdef double x, y, z
    cdef int i

    out = (np.zeros((n_voxel*3, 1))).squeeze()

    for i in range(n_voxel):
        x = coords[i]
        y = coords[i + n_voxel]
        z = coords[i + 2 * n_voxel]

        out[i] = trans_matrix[0] * x + trans_matrix[1] * y + trans_matrix[2] * z + trans_matrix[3]
        out[i + n_voxel] = trans_matrix[4] * x + trans_matrix[5] * y + trans_matrix[6] * z + trans_matrix[7]
        out[i + 2 * n_voxel] = trans_matrix[8] * x + trans_matrix[9] * y + trans_matrix[10] * z + trans_matrix[11]

    return out