# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import CropForeground
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose

TEST_COORDS, TESTS = [], []

for p in TEST_NDARRAYS_ALL:
    TEST_COORDS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[3]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1], "allow_smaller": True},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": [2, 1], "allow_smaller": False},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 4},
            p([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]]),
            p([[[1, 2, 1, 0], [2, 3, 2, 0], [1, 2, 1, 0], [0, 0, 0, 0]]]),
        ]
    )

    TESTS.append(
        [
            {"select_fn": lambda x: x > 0, "channel_indices": None, "margin": 0, "k_divisible": 10},
            p([[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]),
            p(np.zeros((1, 0, 0), dtype=np.int64)),
        ]
    )


class TestCropForeground(unittest.TestCase):
    @parameterized.expand(TEST_COORDS + TESTS)
    def test_value(self, arguments, image, expected_data):
        cropper = CropForeground(**arguments)
        result = cropper(image)
        assert_allclose(result, expected_data, type_test=False)
        self.assertIsInstance(result, MetaTensor)
        self.assertEqual(len(result.applied_operations), 1)
        inv = cropper.inverse(result)
        self.assertIsInstance(inv, MetaTensor)
        self.assertEqual(inv.applied_operations, [])
        self.assertTupleEqual(inv.shape, image.shape)

    @parameterized.expand(TEST_COORDS)
    def test_return_coords(self, arguments, image, _):
        arguments["return_coords"] = True
        _, start_coord, end_coord = CropForeground(**arguments)(image)
        arguments["return_coords"] = False
        np.testing.assert_allclose(start_coord, np.asarray([1, 1]))
        np.testing.assert_allclose(end_coord, np.asarray([4, 4]))


if __name__ == "__main__":
    unittest.main()
