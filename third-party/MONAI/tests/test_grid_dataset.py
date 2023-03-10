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

import sys
import unittest

import numpy as np
from parameterized import parameterized

from monai.data import DataLoader, GridPatchDataset, PatchIter, PatchIterd, iter_patch
from monai.transforms import RandShiftIntensity, RandShiftIntensityd
from monai.utils import set_determinism
from tests.utils import assert_allclose, get_arange_img


def identity_generator(x):
    # simple transform that returns the input itself
    for idx, item in enumerate(x):
        yield item, idx


class TestGridPatchDataset(unittest.TestCase):
    def setUp(self):
        set_determinism(seed=1234)

    def tearDown(self):
        set_determinism(None)

    @parameterized.expand([[True], [False]])
    def test_iter_patch(self, cb):
        shape = (10, 30, 30)
        input_img = get_arange_img(shape)
        for p, _ in iter_patch(input_img, patch_size=(None, 10, 30, None), copy_back=cb):
            p += 1.0
            assert_allclose(p, get_arange_img(shape) + 1.0)
        assert_allclose(input_img, get_arange_img(shape) + (1.0 if cb else 0.0))

    def test_shape(self):
        # test Iterable input data
        test_dataset = iter(["vwxyz", "helloworld", "worldfoobar"])
        result = GridPatchDataset(data=test_dataset, patch_iter=identity_generator, with_coordinates=False)
        output = []
        n_workers = 0 if sys.platform == "win32" else 2
        for item in DataLoader(result, batch_size=3, num_workers=n_workers):
            output.append("".join(item))
        if sys.platform == "win32":
            expected = ["ar", "ell", "ldf", "oob", "owo", "rld", "vwx", "wor", "yzh"]
        else:
            expected = ["d", "dfo", "hel", "low", "oba", "orl", "orl", "r", "vwx", "yzw"]
            self.assertEqual(len("".join(expected)), len("".join(list(test_dataset))))
        self.assertEqual(sorted(output), sorted(expected))

    def test_loading_array(self):
        set_determinism(seed=1234)
        # test sequence input data with images
        images = [np.arange(16, dtype=float).reshape(1, 4, 4), np.arange(16, dtype=float).reshape(1, 4, 4)]
        # image level
        patch_intensity = RandShiftIntensity(offsets=1.0, prob=1.0)
        patch_iter = PatchIter(patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=images, patch_iter=patch_iter, transform=patch_intensity)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
        np.testing.assert_allclose(
            item[0],
            np.array([[[[8.240326, 9.240326], [12.240326, 13.240326]]], [[[10.1624, 11.1624], [14.1624, 15.1624]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(tuple(item[0].shape), (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0],
                np.array(
                    [[[[7.723618, 8.723618], [11.723618, 12.723618]]], [[[10.7175, 11.7175], [14.7175, 15.7175]]]]
                ),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )

    def test_loading_dict(self):
        set_determinism(seed=1234)
        # test sequence input data with dict
        data = [
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
            {
                "image": np.arange(16, dtype=float).reshape(1, 4, 4),
                "label": np.arange(16, dtype=float).reshape(1, 4, 4),
                "metadata": "test string",
            },
        ]
        # image level
        patch_intensity = RandShiftIntensityd(keys="image", offsets=1.0, prob=1.0)
        patch_iter = PatchIterd(keys=["image", "label"], patch_size=(2, 2), start_pos=(0, 0))
        ds = GridPatchDataset(data=data, patch_iter=patch_iter, transform=patch_intensity, with_coordinates=True)
        # use the grid patch dataset
        for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
            np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_equal(item[0]["label"].shape, (2, 1, 2, 2))
            self.assertListEqual(item[0]["metadata"], ["test string", "test string"])
        np.testing.assert_allclose(
            item[0]["image"],
            np.array([[[[8.240326, 9.240326], [12.240326, 13.240326]]], [[[10.1624, 11.1624], [14.1624, 15.1624]]]]),
            rtol=1e-4,
        )
        np.testing.assert_allclose(item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5)
        if sys.platform != "win32":
            for item in DataLoader(ds, batch_size=2, shuffle=False, num_workers=2):
                np.testing.assert_equal(item[0]["image"].shape, (2, 1, 2, 2))
            np.testing.assert_allclose(
                item[0]["image"],
                np.array(
                    [[[[7.723618, 8.723618], [11.723618, 12.723618]]], [[[10.7175, 11.7175], [14.7175, 15.7175]]]]
                ),
                rtol=1e-3,
            )
            np.testing.assert_allclose(
                item[1], np.array([[[0, 1], [2, 4], [0, 2]], [[0, 1], [2, 4], [2, 4]]]), rtol=1e-5
            )


if __name__ == "__main__":
    unittest.main()
