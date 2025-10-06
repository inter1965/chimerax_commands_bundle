"""Rough fitmap command implementation."""

from __future__ import annotations

from .align_center import parse_map_or_atoms


def is_map(session, atomspec):
    from chimerax.map import MapsArg

    maps_arg = MapsArg()
    maps = maps_arg.parse(str(atomspec), session)
    return maps[0] != []


def rough_fitmap(session, atoms_or_map, inmap, search=50, radius=50, sym=False, refine=False):
    from chimerax.atomic import AtomicStructuresArg
    from chimerax.atomic.cmd import combine_cmd
    from chimerax.core.commands import AtomSpecArg, run
    from chimerax.core.commands.cli import command_function
    from chimerax.std_commands import wait

    align_center = command_function("align center")

    atoms_or_map_id = atoms_or_map.spec
    map_id = f"#{inmap[0].id_string}"
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

    a = AtomSpecArg().parse(atoms_or_map_id, session)
    parsed_atoms_or_map = parse_map_or_atoms(session, a[0])

    align_center(session, parsed_atoms_or_map, inmap[0])
    wait.wait(session, 1)

    if sym and not ismap:
        run(session, f"combine {atoms_or_map_id}")
        a = AtomicStructuresArg().parse(atoms_or_map_id, session)
        atoms_or_map = combine_cmd(session, a[0])
        run(session, f"hide {atoms_or_map_id} models")
        atoms_or_map_id = f"#{atoms_or_map.id_string}"

    run(session, f"fitmap {atoms_or_map_id} inmap {map_id} search {search} radius {radius}")
    wait.wait(session, 30)

    if refine:
        run(session, f"fitmap {atoms_or_map_id} inmap {map_id}")

    return atoms_or_map_id, map_id


def rough_fitmap_desc():
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
        ],
        required_arguments=["atoms_or_map", "inmap"],
        synopsis="Initial approximate fitmap command.",
    )


__all__ = ["rough_fitmap", "rough_fitmap_desc"]
