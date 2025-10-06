"""Soft edge mask command implementation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def extend_and_soften_mask(img_in, ini_threshold, extend_ini_mask, width_soft_mask_edge):
    """Return a softened mask based on ``img_in``."""
    img_in = np.asarray(img_in)
    msk_out = np.zeros_like(img_in, dtype=float)

    msk_out[img_in >= ini_threshold] = 1.0

    if extend_ini_mask != 0.0:
        extend_size = abs(extend_ini_mask)
        binary = msk_out.astype(bool)
        if extend_ini_mask > 0:
            distances = distance_transform_edt(~binary)
            msk_out[distances <= extend_size] = 1.0
        else:
            distances = distance_transform_edt(binary)
            msk_out[distances <= extend_size] = 0.0

    if width_soft_mask_edge > 0.0:
        distances = distance_transform_edt(1 - msk_out)
        mask_edge = distances <= width_soft_mask_edge
        mask_soft = np.zeros_like(msk_out)
        mask_soft[mask_edge] = 0.5 + 0.5 * np.cos(
            np.pi * distances[mask_edge] / width_soft_mask_edge
        )
        msk_out[mask_edge] = mask_soft[mask_edge]

    return msk_out


def soft_edge_mask(session, mask, level=0.5, extend=0, width=12):
    from chimerax.map import volume_from_grid_data
    from chimerax.map_data import ArrayGridData

    ini_threshold = level
    extend_ini_mask = extend
    width_soft_edge = width

    session.logger.status(
        f"Binarize map at a threshold level of {ini_threshold:.1f}...",
        log=True,
    )

    if extend_ini_mask != 0.0:
        action = "Extending" if extend_ini_mask > 0 else "Shrinking"
        session.logger.status(
            f"{action} initial binary mask by {extend_ini_mask:.1f}px...",
            log=True,
        )

    if width_soft_edge > 0.0:
        session.logger.status(
            f"Adding a soft edge (of {width_soft_edge:.1f}px) to the mask...",
            log=True,
        )

    input_mask_data = mask[0].data if hasattr(mask, "__iter__") else mask.data
    m = input_mask_data.matrix()
    softmask = extend_and_soften_mask(m, ini_threshold, extend_ini_mask, width_soft_edge)
    new_mask = ArrayGridData(
        softmask,
        origin=input_mask_data.origin,
        step=input_mask_data.step,
        cell_angles=input_mask_data.cell_angles,
        rotation=input_mask_data.rotation,
        symmetries=input_mask_data.symmetries,
    )
    return volume_from_grid_data(new_mask, session)


def soft_edge_mask_desc():
    from chimerax.core.commands import CmdDesc, FloatArg
    from chimerax.map import MapsArg

    return CmdDesc(
        required=[("mask", MapsArg)],
        keyword=[("extend", FloatArg), ("level", FloatArg), ("width", FloatArg)],
        required_arguments=["mask"],
        synopsis="Binarize a map, extend it and apply a raised cosine soft edge.",
    )


__all__ = ["soft_edge_mask", "soft_edge_mask_desc"]
