"""Locate repository-versioned benchmark fixture images."""

from pathlib import Path


def _require_lfs_object(image_path):
    """Raise a clear error when a benchmark image has not been fetched from LFS."""
    with image_path.open("rb") as image_file:
        if image_file.read(len(b"version https://git-lfs")) == b"version https://git-lfs":
            raise FileNotFoundError(
                f"{image_path} is a Git LFS pointer. Fetch the benchmark images with "
                "`git lfs pull --include=\"benchmarks/data/cellclicker/cellcognition_h2b_P0037/**\"`."
            )


def resolve_exported_image_path(images_directory, label_path, position, frame):
    """Resolve a versioned PNG fixture or the original CellCognition TIFF image."""
    images_directory = Path(images_directory)
    bundled_image = images_directory / f"P{position}_t{frame:03d}.png"
    if bundled_image.is_file():
        _require_lfs_object(bundled_image)
        return bundled_image

    raw_image = images_directory / position / f"tubulin_P{position}_T{frame:05}_Crfp_Z1_S1.tif"
    if raw_image.is_file():
        return raw_image
    raise FileNotFoundError(
        f"No image for {Path(label_path).name}. Expected {bundled_image} or {raw_image}."
    )
