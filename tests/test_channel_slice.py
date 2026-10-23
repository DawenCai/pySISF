"""Archive reads of a channel range that does not start at channel 0."""

import numpy as np

from pySISF import sisf


def test_channel_slice_not_starting_at_zero(tmp_path):
    data = np.stack([np.full((20, 12, 8), c + 1, np.uint16) for c in range(3)])  # [c, x, y, z]
    root = str(tmp_path / "a")
    sisf.create_sisf(root, data, (20, 12, 8), (8, 8, 8), (100, 100, 100), enable_status=False,
                     thread_count=1)
    arc = sisf.sisf(root)
    assert np.all(arc[1:3, :, :, :][0] == 2) and np.all(arc[1:3, :, :, :][1] == 3)
    assert np.all(arc[2, 0:4, 0:4, 0:4] == 3)
    assert np.array_equal(arc[0:3, :, :, :], data)
