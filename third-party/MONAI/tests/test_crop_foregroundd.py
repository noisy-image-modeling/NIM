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

from monai.transforms import CropForegroundd
from tests.utils import TEST_NDARRAYS_ALL, assert_allclose

TEST_POSITION, TESTS = [], []
for p in TEST_NDARRAYS_ALL:
    TEST_POSITION.append(
        [
            {
                "keys": ["img", "label"],
                "source_key": "label",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": 0,
            },
            {
                "img": p(
                    np.array([[[1, 0, 2, 0, 1], [0, 1, 2, 1, 0], [2, 2, 3, 2, 2], [0, 1, 2, 1, 0], [1, 0, 2, 0, 1]]])
                ),
                "label": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]])
                ),
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 1, "channel_indices": None, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[3]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": 0, "margin": 0},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])),
        ]
    )
    TESTS.append(
        [
            {"keys": ["img"], "source_key": "img", "select_fn": lambda x: x > 0, "channel_indices": None, "margin": 1},
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 0, 0, 0, 0]]])),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": True,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": None,
                "margin": [2, 1],
                "allow_smaller": False,
            },
            {
                "img": p(
                    np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])
                )
            },
            p(
                np.array(
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 2, 1, 0],
                            [0, 2, 3, 2, 0],
                            [0, 1, 2, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ]
                )
            ),
        ]
    )
    TESTS.append(
        [
            {
                "keys": ["img"],
                "source_key": "img",
                "select_fn": lambda x: x > 0,
                "channel_indices": 0,
                "margin": 0,
                "k_divisible": [4, 6],
                "mode": "edge",
            },
            {
                "img": p(
                    np.array(
                        [[[0, 2, 1, 2, 0], [1, 1, 2, 1, 1], [2, 2, 3, 2, 2], [1, 1, 2, 1, 1], [0, 0, 0, 0, 0]]],
                        dtype=np.float32,
                    )
                )
            },
            p(np.array([[[0, 2, 1, 2, 0, 0], [1, 1, 2, 1, 1, 1], [2, 2, 3, 2, 2, 2], [1, 1, 2, 1, 1, 1]]])),
        ]
    )


class TestCropForegroundd(unittest.TestCase):
    @parameterized.expand(TEST_POSITION + TESTS)
    def test_value(self, arguments, input_data, expected_data):
        cropper = CropForegroundd(**arguments)
        result = cropper(input_data)
        assert_allclose(result["img"], expected_data, type_test="tensor")
        if "label" in input_data and "img" in input_data:
            self.assertTupleEqual(result["img"].shape, result["label"].shape)
        inv = cropper.inverse(result)
        self.assertTupleEqual(inv["img"].shape, input_data["img"].shape)
        if "label" in input_data:
            self.assertTupleEqual(inv["label"].shape, input_data["label"].shape)

    @parameterized.expand(TEST_POSITION)
    def test_foreground_position(self, arguments, input_data, _):
        result = CropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["foreground_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["foreground_end_coord"], np.array([4, 4]))

        arguments["start_coord_key"] = "test_start_coord"
        arguments["end_coord_key"] = "test_end_coord"
        result = CropForegroundd(**arguments)(input_data)
        np.testing.assert_allclose(result["test_start_coord"], np.array([1, 1]))
        np.testing.assert_allclose(result["test_end_coord"], np.array([4, 4]))


if __name__ == "__main__":
    unittest.main()
