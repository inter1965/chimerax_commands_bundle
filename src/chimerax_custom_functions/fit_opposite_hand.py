"""Fit opposite hand command implementation."""

from __future__ import annotations

import numpy as np

from .rough_fitmap import is_map


def fit_opposite_hand(
    session,
    atoms_or_map,
    inmap,
    search=50,
    radius=50,
    sym=False,
    refine=True,
    SkipRoughFit=False,
):
    from chimerax.atomic import AtomicStructuresArg
    from chimerax.atomic.cmd import combine_cmd
    from chimerax.core.commands import run
    from chimerax.map_filter.vopcommand import volume_flip

    map_id = f"#{inmap[0].id_string}"
    atoms_or_map_id = atoms_or_map.spec
    ismap = is_map(session, atoms_or_map.spec)

    if inmap[0]._surfaces[0]._colors[0][3] == 255:
        run(session, f"trans {map_id} 70")

    if sym and not ismap:
        orig_models = session.models._models.copy()
        run(session, f"sym {atoms_or_map_id} biomt")
        dif = session.models._models.keys() - orig_models
        new_ids = [key for key in dif if len(key) == 1]
        if new_ids:
            atoms_or_map_id = "#" + str(new_ids[0][0])

    flipped_volume = volume_flip(session, inmap)
    flipped_volume_id = f"#{flipped_volume.id_string}"

    run(session, f"combine {atoms_or_map_id}")
    combined = AtomicStructuresArg().parse(atoms_or_map_id, session)
    atoms_or_map = combine_cmd(session, combined[0])

    run(session, f"hide {atoms_or_map_id} models")
    old_atoms_or_map_id = atoms_or_map_id
    atoms_or_map_id = f"#{atoms_or_map.id_string}"

    map_size = np.array(inmap[0].data.size)
    map_step = inmap[0].data.step
    map_center = (map_size / 2.0) * map_step
    center_str = "%.5g,%.5g,%.5g" % tuple(map_center)
    run(
        session,
        f"turn x 180 coordinateSystem {flipped_volume_id} center {center_str} models {atoms_or_map_id}",
    )

    if not SkipRoughFit:
        run(
            session,
            f"rough fitmap {atoms_or_map_id} inmap {flipped_volume_id} search {search} radius {radius} refine False",
        )

    if refine:
        run(session, f"fitmap {atoms_or_map_id} inmap {flipped_volume_id}")

    if SkipRoughFit and not refine:
        session.logger.status("No fitting done. Set refine True or SkipRoughFit False.", log=True)

    cmd1 = f"hide {atoms_or_map_id} models; hide {flipped_volume_id} models; show {old_atoms_or_map_id} models; show {map_id} models"
    cmd2 = f"hide {old_atoms_or_map_id} models; hide {map_id} models; show {atoms_or_map_id} models; show {flipped_volume_id} models"
    session.logger.status("To view original hand fit, run command:", log=True)
    session.logger.status(cmd1, log=True)
    session.logger.status("To view opposite hand fit, run command:", log=True)
    session.logger.status(cmd2, log=True)

    return atoms_or_map_id, flipped_volume_id


def fit_opposite_hand_desc():
    from chimerax.core.commands import BoolArg, CmdDesc, IntArg, ObjectsArg
    from chimerax.map import MapsArg

    return CmdDesc(
        required=[("atoms_or_map", ObjectsArg)],
        keyword=[
            ("inmap", MapsArg),
            ("search", IntArg),
            ("radius", IntArg),
            ("sym", BoolArg),
            ("refine", BoolArg),
            ("SkipRoughFit", BoolArg),
        ],
        required_arguments=["atoms_or_map", "inmap"],
        synopsis="Fit a model into an opposite-hand map.",
    )


__all__ = ["fit_opposite_hand", "fit_opposite_hand_desc"]
