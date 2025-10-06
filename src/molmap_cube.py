"""Molmap cube command implementation."""

from __future__ import annotations

import numpy as np

from .align_center import define_centroid


def molmap_cube(session, atoms, resolution, size, spacing):
    from chimerax.core.commands import run
    from chimerax.map import molmap
    from chimerax.map_filter.vopcommand import volume_new

    box = volume_new(
        session,
        name="box",
        size=(size, size, size),
        grid_spacing=(spacing, spacing, spacing),
    )
    box_id = box.id_string

    run(session, f"volume #{box_id} level 0")
    run(session, f"trans #{box_id} 70")

    atoms_center = define_centroid(session, atoms)
    half_map = (size / 2.0) * spacing
    map_center = (half_map, half_map, half_map)

    s2c = session.main_view.camera.position.inverse()
    screen_map_center = s2c.transform_points(np.expand_dims(map_center, 0))
    screen_atoms_center = s2c.transform_points(np.expand_dims(atoms_center, 0))
    center_dif = [-1 * (m - a) for m, a in zip(screen_map_center[0], screen_atoms_center[0])]
    run(session, "move %0.2f,%0.2f,%0.2f models #%s" % (
        center_dif[0],
        center_dif[1],
        center_dif[2],
        box_id,
    ))

    session.logger.status("Running molmap with onGrid option...", log=True)
    molmap.molmap(session, atoms, resolution, on_grid=box)
    session.logger.status("Done.", log=True)
    session.logger.status(
        "Box displayed is %d pixels with a spacing of %.2f angstrom/pixel" % (size, spacing),
        log=True,
    )


def molmap_cube_desc():
    from chimerax.atomic import AtomsArg
    from chimerax.core.commands import CmdDesc, FloatArg, IntArg

    return CmdDesc(
        required=[("atoms", AtomsArg), ("resolution", FloatArg), ("size", IntArg), ("spacing", FloatArg)],
        keyword=[],
        required_arguments=["atoms", "resolution", "size", "spacing"],
        synopsis="Create a cubic molmap volume with specified size and spacing.",
    )


__all__ = ["molmap_cube", "molmap_cube_desc"]
