"""
Preprocessing layer for image augmentation in neuronal networks based on the NiftyNet framework.

The NiftyNet Framework (https://niftynet.io/) is Apache licensed:

        Copyright 2018 University College London and the NiftyNet Contributors

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.

The layer combines several different transformations for image augmentation.
The structure of this class has been adopted from NiftyNet to make it compatible with the framework.

The present code itself is licensed through the BSD-3 license:

        BSD 3-Clause License

        Copyright (c) 2020, aaronbacher
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this
           list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice,
           this list of conditions and the following disclaimer in the documentation
           and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
           contributors may be used to endorse or promote products derived from
           this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
from scipy.ndimage.interpolation import map_coordinates, geometric_transform, affine_transform
from scipy.ndimage.filters import gaussian_filter

import SimpleITK as sitk	# for debugging only
import os			# for debugging only

from niftynet_additions import transform_cy

from niftynet.layer.base_layer import RandomisedLayer

def translation_3d(x, y, z):
    trans_matrix = np.array([[1.0, 0.0, 0.0, x],
                             [0.0, 1.0, 0.0, y],
                             [0.0, 0.0, 1.0, z],
                             [0.0, 0.0, 0.0, 1.0]])

    return trans_matrix


class RandomTransformationLayer(RandomisedLayer):
    """
    apply different transformations (rotation, scaling, elastic deformation)
    """

    def __init__(self,
                 do_flip=True,
                 do_scale=True,
                 do_rotate=True,
                 do_elastic=False,
                 flip_axes=(1, 0),
                 flip_probability=0,
                 lr_labels=None,
                 min_percentage=0,
                 max_percentage=0,
                 antialiasing=True,
                 isotropic=False,
                 elastic_sigma=4,
                 elastic_alpha=0,
                 name='random_transformation'):
        """
        :param do_flip: bool, choose if flips should be performed or not
        :param do_scale: bool, choose if scalings should be performed or not
        :param do_rotate: bool, choose if rotations should be performed or not
        :param do_elastic: bool, choose if elastic deformation should be performed or not

        flipping
        :param flip_axes: a list of indices over which to flip
        :param flip_probability: the probability of performing the flip (default = 0.5)
        :param lr_labels: list of tuples of right and left labels of organs that need to be switched in case of flip
            example: [(left_organ_x_label, right_organ_x_label), (left_organ_y_label, right_organ_y_label)]

        scaling
        :param min_percentage: minimal percantage to scale
        :param max_percentage: maximal percantage to scale
        :param antialiasing: bool, choose if antialiasing should be performed or not
        :param isotropic: bool, true if scaling should be the same in every direction

        elastic deformataion
        elastic deformation according to implementation of Simard et al.
        (Simard, Patrice Y., David Steinkraus, and John C. Platt. "Best practices for convolutional neural networks applied to visual document analysis." Icdar. Vol. 3. No. 2003. 2003.)
        :param elastic_sigma: value for elasticity coefficient
        :param elastic_alpha: impact of elastic deformation

        general
        :param name: name/description of layer
        """

        super(RandomTransformationLayer, self).__init__(name=name)
        self._transform = None  # transform-matrix
        self._combined = False  # Flag to tell if single transformations have already been combined

        # elastic
        if do_elastic:
            self._do_elastic = True
            self._elastic_seed = None
            self._elastic_sigma = elastic_sigma
            self._elastic_alpha = elastic_alpha
            self._dx = None
            self._dy = None
            self._dz = None
        else:
            self._do_elastic = False

        # rotating
        if do_rotate:
            self._do_rotation = True
            self._rotation = None  # rotation-matrix
            self.min_angle = None
            self.max_angle = None
            self.rotation_angle_x = None
            self.rotation_angle_y = None
            self.rotation_angle_z = None
        else:
            self._do_rotation = False

        # flipping
        if do_flip:
            self._do_flip = True
            self._flip = None  # flip-matrix
            self._flip_axes = flip_axes
            self._flip_probability = flip_probability
            self._rand_flip = None
            self._num_flips = 0
            self._lr_labels = []
            self._swap_labels = False

            for char in lr_labels:
                if not char.isnumeric():
                    lr_labels = lr_labels.replace(char, '')

            assert len(lr_labels) % 2 == 0, "number of arguments passed " \
                                            "to lr_labels in config has to be even!"

            for i, label in enumerate(lr_labels):
                if i % 2 == 0:
                    self._lr_labels.append((int(label), int(lr_labels[i + 1])))

        else:
            self._do_flip = False

        # scaling
        if do_scale:
            self._do_scale = True
            self._rand_zoom = None
            self._scale = None
            assert min_percentage <= max_percentage
            self._min_percentage = max(min_percentage, -99.9)
            self._max_percentage = max_percentage
            self._isotropic = isotropic
            self.antialiasing = antialiasing
        else:
            self._do_scale = False

    def init_uniform_angle(self, rotation_angle=(-10.0, 10.0)):
        if rotation_angle == 0 or rotation_angle == (0, 0):
            self._do_rotation = False
            return

        assert rotation_angle[0] < rotation_angle[1]
        self.min_angle = float(rotation_angle[0])
        self.max_angle = float(rotation_angle[1])

    def init_non_uniform_angle(self,
                               rotation_angle_x,
                               rotation_angle_y,
                               rotation_angle_z):
        if len(rotation_angle_x):
            assert rotation_angle_x[0] < rotation_angle_x[1]
        if len(rotation_angle_y):
            assert rotation_angle_y[0] < rotation_angle_y[1]
        if len(rotation_angle_z):
            assert rotation_angle_z[0] < rotation_angle_z[1]
        self.rotation_angle_x = [float(e) for e in rotation_angle_x]
        self.rotation_angle_y = [float(e) for e in rotation_angle_y]
        self.rotation_angle_z = [float(e) for e in rotation_angle_z]

    def randomise(self, spatial_rank=3):
        # Rotation
        if spatial_rank == 3:
            self._rotation_3d()
        else:
            # currently not supported spatial rank for rand rotation
            pass

        # Flip
        self._rand_flip = np.random.random(size=int(np.floor(spatial_rank))) < self._flip_probability
        self._flip_3d()

        # Scale
        if self._do_scale:
            if self._isotropic:
                one_rand_zoom = np.random.uniform(low=self._min_percentage, high=self._max_percentage)
                rand_zoom = np.repeat(one_rand_zoom, spatial_rank)
            else:
                rand_zoom = np.random.uniform(low=self._min_percentage, high=self._max_percentage, size=(int(np.floor(spatial_rank)),))
            self._rand_zoom = (rand_zoom + 100.0)/100.0

            self._scale_3d()
        else:
            self._scale = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])

    def _rotation_3d(self):

        if not self._do_rotation:
            self._rotation = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
            return

        angle_x = 0.0
        angle_y = 0.0
        angle_z = 0.0
        if self.min_angle is None and self.max_angle is None:
            # generate transformation
            if len(self.rotation_angle_x) >= 2:
                angle_x = np.random.uniform(
                    self.rotation_angle_x[0],
                    self.rotation_angle_x[1]) * np.pi / 180.0

            if len(self.rotation_angle_y) >= 2:
                angle_y = np.random.uniform(
                    self.rotation_angle_y[0],
                    self.rotation_angle_y[1]) * np.pi / 180.0

            if len(self.rotation_angle_z) >= 2:
                angle_z = np.random.uniform(
                    self.rotation_angle_z[0],
                    self.rotation_angle_z[1]) * np.pi / 180.0
        else:
            # generate transformation
            angle_x = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
            angle_y = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
            angle_z = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0

        # we create already inverted matrices, so they haven't to be inverted later
        rotation_x = np.array([[np.cos(angle_x), np.sin(angle_x), 0.0, 0.0],
                               [-np.sin(angle_x), np.cos(angle_x), 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

        rotation_y = np.array([[np.cos(angle_y), 0.0, -np.sin(angle_y), 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [np.sin(angle_y), 0.0, np.cos(angle_y), 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

        rotation_z = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, np.cos(angle_z), np.sin(angle_z), 0.0],
                               [0.0, -np.sin(angle_z), np.cos(angle_z), 0.0],
                               [0.0, 0.0, 0.0, 1.0]])

        rotation = np.dot(rotation_z, np.dot(rotation_x, rotation_y))

        self._rotation = rotation

    def _flip_3d(self):
        flip = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

        if self._flip_axes == -1:
            self._do_flip = False
            self._flip = flip
            return

        assert self._rand_flip is not None, "Flip is unset -- Error!"

        # again create inverted matrices (but they are the same as not inverted ^^)
        for axis_number, do_flip in enumerate(self._rand_flip):
            if axis_number in self._flip_axes and do_flip:
                if axis_number == 0:
                    flip_x = np.array([[-1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]])
                    flip = np.dot(flip, flip_x)
                    self._num_flips += 1
                if axis_number == 1:
                    flip_y = np.array([[1.0, 0.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]])
                    flip = np.dot(flip, flip_y)
                    self._num_flips += 1
                if axis_number == 2:
                    flip_z = np.array([[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, -1.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]])
                    flip = np.dot(flip, flip_z)
                    self._num_flips += 1
        self._flip = flip
        if self._num_flips % 2 == 1:    # labels have to be swapped only if amount of flips is odd
            self._swap_labels = True

    def _change_rl_labels(self, image):
        max_label = np.amax(image)
        for i in self._lr_labels:
            image = np.where(image == i[0], (max_label + 1), image)
            image = np.where(image == i[1], i[0], image)
            image = np.where(image == (max_label + 1), i[1], image)
        return image

    def _scale_3d(self):
        assert len(self._rand_zoom) == 3, 'Scaling in 3 dimensions only!'
        x = self._rand_zoom[0]
        y = self._rand_zoom[1]
        z = self._rand_zoom[2]

        # again create inverted matrix
        scale = np.array([[1/x, 0.0, 0.0, 0.0],
                          [0.0, 1/y, 0.0, 0.0],
                          [0.0, 0.0, 1/z, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

        self._scale = scale

    def _combine_transformations(self, w, h, d):

        trans1 = translation_3d(-w/2, -h/2, -d/2)
        trans2 = translation_3d(w/2, h/2, d/2)

        self._transform = np.dot(self._rotation, np.dot(self._scale, self._flip))
        self._transform = np.dot(trans2, self._transform)
        self._transform = np.dot(self._transform, trans1)

        self._combined = True

    def _get_antialiasing_sigma(self, zoom):
        """
        Compute optimal standard deviation for Gaussian kernel.

            Cardoso et al., "Scale factor point spread function matching:
            beyond aliasing in image resampling", MICCAI 2015
        """
        k = 1 / zoom
        variance = (k ** 2 - 1 ** 2) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma = np.sqrt(variance)
        return sigma

    def _apply_transformation_3d(self, image_3d, field_is_label, interp_order=3):

        if interp_order < 0:
            return image_3d

        assert all([dim > 1 for dim in image_3d.shape]), 'random rotation supports 3D inputs only'

        img = image_3d
        # Antialiasing Filter
        if self._do_scale:
            assert self._rand_zoom is not None
            full_zoom = np.array(self._rand_zoom)
            while len(full_zoom) < image_3d.ndim:
                full_zoom = np.hstack((full_zoom, [1.0]))
            is_undersampling = all(full_zoom[:3] < 1)
            run_antialiasing_filter = self.antialiasing and is_undersampling

            if run_antialiasing_filter:
                sigma = self._get_antialiasing_sigma(full_zoom[:3])

                if image_3d.ndim == 4:
                    print("Warning: image with dimension 4 not tested yet!")
                    for mod in range(image_3d.shape[-1]):
                        img = gaussian_filter(image_3d[..., mod], sigma)
                elif image_3d.ndim == 3:
                    img = gaussian_filter(image_3d, sigma)
                else:
                    raise NotImplementedError('not implemented random scaling')

        # combine all transformations
        if not self._combined:
            w, h, d = img.shape
            self._combine_transformations(w, h, d)
        assert self._transform is not None, 'No transformation-matrix created'

        # execute transformation
        assert (len(img.shape) == 3), 'image does not have 3 dimensions'

        """ elastic transforamtion"""
        # depending if elastic deformation is asked for or not, image transformation will be executed in one way or another

        # with elastic deformation:
        if self._do_elastic:
            shape = img.shape
            # check only one, if dx is None, dy and dz are None as well
            if self._dx is None:
                random_state = np.random.RandomState(None)
                self._dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self._elastic_sigma, mode="constant",
                                           cval=0) * self._elastic_alpha
                self._dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self._elastic_sigma, mode="constant",
                                           cval=0) * self._elastic_alpha
                self._dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), self._elastic_sigma, mode="constant",
                                           cval=0) * self._elastic_alpha

            x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + self._dx, (-1, 1)), \
                      np.reshape(y + self._dy, (-1, 1)), \
                      np.reshape(z + self._dz, (-1, 1))

            # be sure there are the same amount of coordinates in each
            assert (indices[0].shape[0] == indices[1].shape[0] == indices[2].shape[0]), "Something went wrong"

            """ affine transformation"""
            self._transform = (self._transform.reshape((1, 16))).squeeze()

            indices = np.concatenate((indices[0], indices[1], indices[2])).squeeze()
            indices_out = transform_cy.affine_transformation_cy(indices, (int)(indices.shape[0] / 3), self._transform)
            indices_out = indices_out.reshape(3, (shape[0] * shape[1] * shape[2])).T
            indices_final = np.reshape(indices_out[:, 0], (-1, 1)), np.reshape(indices_out[:, 1], (-1, 1)), np.reshape(
                indices_out[:, 2], (-1, 1))

            img = map_coordinates(img, indices_final, order=interp_order).reshape(shape)

            if field_is_label and self._swap_labels:   # change labels only if number flips is odd
                img = self._change_rl_labels(img)

            return img

        else:   # no elastic deformation

            if not self._do_scale and not self._do_rotation and self._do_flip:
                for ax, do_flip in enumerate(self._rand_flip):
                    if do_flip:
                        img = np.flip(img, axis=ax)
            else:
                affine_matrix = self._transform[0:3, 0:3]
                center_ = 0.5 * np.asarray(image_3d.shape, dtype=np.int64)
                c_offset = center_ - center_.dot(affine_matrix)
                img = affine_transform(img, affine_matrix.T, c_offset, order=interp_order)

            if field_is_label and self._swap_labels:
                img = self._change_rl_labels(img)

            return img

    def layer_op(self, inputs, interp_orders, *args, **kwargs):

        debug = False
        safe_path = os.getcwd() + "/../../my_transform_debug/"  # for debugging

        if inputs is None:
            return inputs

        if debug:
            for field, image in inputs.items():
                # write image
                image_to_safe = sitk.GetImageFromArray(image.copy().squeeze())
                sitk.WriteImage(image_to_safe, safe_path + "myCnn_" + str(field) + "_input.nrrd")

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                # from rand_elastic_deform
                assert image.shape[-1] == len(interp_orders[field]), \
                    "interpolation orders should be" \
                    "specified for each inputs modality"

                # from rand_rotation
                interp_order = interp_orders[field][0]

                # from rand_flip
                assert (all([i < 0 for i in interp_orders[field]]) or
                        all([i >= 0 for i in interp_orders[field]])), \
                    'Cannot combine interpolatable and non-interpolatable data'
                if field == 'label':
                    field_is_label = True
                else:
                    field_is_label = False

                if interp_orders[field][0] < 0:
                    continue

                if image.ndim == 3:
                    print("Should not occur, for debug only!")
                    inputs[field] = self._apply_transformation_3d(image[...], field_is_label, interp_order)
                # continue with standart execution (present in all single layers)
                else:
                    for channel_idx in range(image.shape[-1]):
                        if image.ndim == 4:
                            inputs[field][..., channel_idx] = \
                                self._apply_transformation_3d(image[..., channel_idx], field_is_label, interp_order)
                        elif image.ndim == 5:
                            for t in range(image.shape[-2]):
                                inputs[field][..., t, channel_idx] = \
                                    self._apply_transformation_3d(image[..., t, channel_idx], field_is_label, interp_order)

                        else:
                            raise NotImplementedError("unknown input format")

        else:
            raise NotImplementedError("unknown input format")

        # ----------------------------------------------------------------------------
        # DEBUG
        if debug:
            with open(safe_path + "info.txt", "w") as file:

                file.write("-------------------------\nVariables:\n")
                variables = vars(self)
                for var_name in vars(self):
                    file.write(var_name + ": " + str(variables[var_name]) + "\n")

                file.write("\n\n-------------------------\nInputs:\n")
                for field, image in inputs.items():
                    # document input
                    file.write("name: " + field + "\tinterp_order: " + str(interp_orders[field][0]) + "\n")

                    # write image
                    image_to_safe = sitk.GetImageFromArray(image.squeeze())
                    sitk.WriteImage(image_to_safe, safe_path + "myCnn_" + str(field) + "_output.nrrd")

        return inputs
