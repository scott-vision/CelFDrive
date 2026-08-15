from pathlib import Path
import re

import numpy as np
import tifffile


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "max_projection_sahi"


def test_example_contains_complete_uint16_position_z_grid_and_exact_projections():
    """Every released max projection must equal its three named Z planes."""
    pattern = re.compile(r"p(?P<position>[1-4])z(?P<z>[1-3])\.tif$")
    slice_paths = {}
    for path in (EXAMPLE_ROOT / "z_slices").glob("*.tif"):
        match = pattern.fullmatch(path.name)
        if match:
            slice_paths[(int(match["position"]), int(match["z"]))] = path

    expected = {(position, z_index) for position in range(1, 5) for z_index in range(1, 4)}
    assert set(slice_paths) == expected
    for position in range(1, 5):
        planes = [tifffile.imread(slice_paths[(position, z_index)]) for z_index in range(1, 4)]
        assert all(plane.shape == (2304, 2304) and plane.dtype == np.uint16 for plane in planes)
        projection = tifffile.imread(EXAMPLE_ROOT / "max_projections" / f"p{position}_max.tif")
        assert projection.shape == (2304, 2304)
        assert projection.dtype == np.uint16
        assert np.array_equal(projection, np.max(np.stack(planes, axis=0), axis=0))
