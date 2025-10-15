"""Utilities for packing UV-grid NPZ files into BrepDiff-compatible HDF5 sets.

The converter reads the NPZ files produced by :mod:`scripts.visualization.vis_step`
or any other :meth:`brepdiff.primitives.uvgrid.UvGrid.export_npz` call and writes
``coords``, ``normals``, ``masks`` and ``types`` datasets under ``data/<uid>`` in
the destination HDF5 file. Optionally it can also emit a newline separated list
of the stored sample identifiers to bootstrap custom dataset splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import numpy as np
import typer

from brepdiff.primitives.uvgrid import UvGrid

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class PackedSample:
    """Container for the datasets written to the HDF5 file."""

    coords: np.ndarray
    normals: np.ndarray
    masks: np.ndarray
    types: np.ndarray


def _collect_npz_paths(npz_root: Path, pattern: str) -> List[Path]:
    if npz_root.is_file():
        return [npz_root]

    if not npz_root.is_dir():
        raise typer.BadParameter(f"{npz_root} is neither a file nor a directory")

    npz_paths = sorted(npz_root.glob(pattern))
    return npz_paths


def _strip_suffix(name: str, suffix: str) -> str:
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def _load_sample(npz_path: Path, drop_empty: bool, fill_value: int) -> PackedSample:
    npz_data = dict(np.load(npz_path))
    uvgrid = UvGrid.load_from_npz_data(npz_data)

    if drop_empty and getattr(uvgrid, "empty_mask", None) is not None:
        keep_mask = ~uvgrid.empty_mask
        if keep_mask.ndim != 1:
            raise ValueError(
                f"Expected 1D empty_mask for {npz_path.name}, got shape {keep_mask.shape}"
            )
        uvgrid.coord = uvgrid.coord[keep_mask]
        if uvgrid.grid_mask is not None:
            uvgrid.grid_mask = uvgrid.grid_mask[keep_mask]
        if uvgrid.normal is not None:
            uvgrid.normal = uvgrid.normal[keep_mask]
        if uvgrid.prim_type is not None:
            uvgrid.prim_type = uvgrid.prim_type[keep_mask]

    coords = np.asarray(uvgrid.coord, dtype=np.float32)
    if coords.ndim != 4:
        raise ValueError(
            f"Expected coords with 4 dimensions (n_prims, res, res, 3) in {npz_path.name},"
            f" got shape {coords.shape}"
        )

    if uvgrid.normal is None:
        normals = np.zeros_like(coords, dtype=np.float32)
    else:
        normals = np.asarray(uvgrid.normal, dtype=np.float32)

    if uvgrid.grid_mask is None:
        masks = np.linalg.norm(coords, axis=-1) > 1e-6
    else:
        masks = np.asarray(uvgrid.grid_mask, dtype=np.bool_)

    types = (
        np.asarray(uvgrid.prim_type, dtype=np.int32)
        if uvgrid.prim_type is not None
        else np.full((coords.shape[0],), fill_value, dtype=np.int32)
    )

    return PackedSample(coords=coords, normals=normals, masks=masks, types=types)


def _write_sample(
    data_group: h5py.Group,
    uid: str,
    sample: PackedSample,
    overwrite: bool,
    compression: Optional[str],
) -> None:
    if uid in data_group:
        if not overwrite:
            raise FileExistsError(
                f"Sample '{uid}' already exists; rerun with --overwrite to replace it"
            )
        del data_group[uid]

    entry = data_group.create_group(uid)
    entry.create_dataset("coords", data=sample.coords, compression=compression)
    entry.create_dataset("normals", data=sample.normals, compression=compression)
    entry.create_dataset("masks", data=sample.masks.astype(np.bool_), compression=compression)
    entry.create_dataset("types", data=sample.types.astype(np.int32), compression=compression)


def _write_split_file(list_path: Path, uids: Iterable[str]) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\n")


@app.command()
def convert(
    npz_root: Path = typer.Argument(..., help="Path to a uvgrid NPZ file or directory"),
    output_h5: Path = typer.Argument(..., help="Destination HDF5 file"),
    list_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to write the newline-separated uid list for the dataset",
    ),
    pattern: str = typer.Option(
        "*_uvgrid.npz",
        help="Glob pattern used when npz_root is a directory",
    ),
    uid_suffix: str = typer.Option(
        "_uvgrid",
        help="Suffix removed from each file stem to form the uid",
    ),
    drop_empty: bool = typer.Option(
        True,
        help="Remove primitives flagged as empty before writing",
    ),
    overwrite: bool = typer.Option(
        True,
        help="Replace existing samples when the uid already exists in the H5 file",
    ),
    compress: bool = typer.Option(
        True,
        help="Store datasets with gzip compression to save disk space",
    ),
    default_type: int = typer.Option(
        1,
        help="Primitive label to use when the NPZ file does not provide prim_type",
    ),
):
    """Pack NPZ UV-grids into an HDF5 file compatible with BrepDiff datasets."""

    npz_paths = _collect_npz_paths(npz_root, pattern)
    if not npz_paths:
        typer.echo("No NPZ files found. Nothing to convert.")
        raise typer.Exit(code=1)

    compression = "gzip" if compress else None
    uids: List[str] = []

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "a") as h5_file:
        data_group = h5_file.require_group("data")

        for npz_path in npz_paths:
            uid = _strip_suffix(npz_path.stem, uid_suffix)
            sample = _load_sample(npz_path, drop_empty=drop_empty, fill_value=default_type)
            _write_sample(
                data_group,
                uid,
                sample,
                overwrite=overwrite,
                compression=compression,
            )
            uids.append(uid)
            typer.echo(f"Stored {uid} in {output_h5}")

    if list_path is not None:
        _write_split_file(list_path, uids)
        typer.echo(f"Wrote {len(uids)} entries to {list_path}")

    typer.echo(f"Finished packing {len(uids)} sample(s) into {output_h5}")


if __name__ == "__main__":
    app()
