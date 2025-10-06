"""Align center command implementation."""

from __future__ import annotations

import numpy as np


def define_centroid(session, atoms, mass_weighting=False):
    from chimerax.atomic import AtomicStructure, concatenate
    from chimerax.centroids import centroid
    from chimerax.core.errors import UserError

    if atoms is None:
        structures_atoms = [m.atoms for m in session.models if isinstance(m, AtomicStructure)]
        if structures_atoms:
            atoms = concatenate(structures_atoms)
    if not atoms:
        raise UserError("Atom specifier selects no atoms")
    structures = atoms.unique_structures
    crds = atoms.scene_coords if len(structures) > 1 else atoms.scene_coords
    if mass_weighting:
        masses = atoms.elements.masses
        avg_mass = masses.sum() / len(masses)
        weights = masses[:, np.newaxis] / avg_mass
    else:
        weights = None
    xyz = centroid(crds, weights=weights)
    return xyz


def parse_map_or_atoms(session, atomspec):
    from chimerax.atomic import AtomsArg
    from chimerax.map import MapsArg

    maps_arg = MapsArg()
    maps = maps_arg.parse(str(atomspec), session)
    if maps[0] == []:
        atoms_arg = AtomsArg()
        atoms = atoms_arg.parse(str(atomspec), session)
        return atoms[0]
    return maps[0][0]


def align_center(session, model, to=None, MoveAtomSubset=False):
    from chimerax.atomic.molarray import Atoms
    from chimerax.core.commands import atomspec, run
    from chimerax.map.volume import Volume
    from chimerax.std_commands.measure_center import volume_center_of_mass

    if isinstance(model, atomspec.AtomSpec):
        model = parse_map_or_atoms(session, model)
    if to is not None and isinstance(to, atomspec.AtomSpec):
        to = parse_map_or_atoms(session, to)

    if isinstance(model, Atoms):
        model_id = model.spec
        model_center = define_centroid(session, model)
        move_string = "atoms"
    elif isinstance(model, Volume):
        model_id = f"#{model.id_string}"
        model_center = volume_center_of_mass(model)
        if np.isnan(model_center[0]):
            raise ValueError("Map has no volume. Set threshold level to display density.")
        model_xyz = model.data.ijk_to_xyz(model_center)
        model_center = model.scene_position * model_xyz
        move_string = "models"
    else:
        raise ValueError(f"Model type not recognised: {type(model)}")

    if to is not None:
        if isinstance(to, Atoms):
            to_center = define_centroid(session, to)
        elif isinstance(to, Volume):
            to_center = volume_center_of_mass(to)
            if np.isnan(to_center[0]):
                raise ValueError("Map has no volume. Set threshold level to display density.")
            to_xyz = to.data.ijk_to_xyz(to_center)
            to_center = to.scene_position * to_xyz
        else:
            raise ValueError(f"'Model to' type not recognised: {type(to)}")
    else:
        to_center = session.main_view.center_of_rotation

    move_atom_subset = MoveAtomSubset
    if not move_atom_subset:
        move_string = "model"

    s2c = session.main_view.camera.position.inverse()
    screen_model_center = s2c.transform_points(np.expand_dims(model_center, 0))
    screen_target_center = s2c.transform_points(np.expand_dims(to_center, 0))
    center_dif = [-1 * (m - a) for m, a in zip(screen_model_center[0], screen_target_center[0])]
    cmd = "move %0.2f,%0.2f,%0.2f %s %s" % (
        center_dif[0],
        center_dif[1],
        center_dif[2],
        move_string,
        model_id,
    )
    run(session, cmd)


def align_center_desc():
    from chimerax.core.commands import AtomSpecArg, BoolArg, CmdDesc

    return CmdDesc(
        required=[("model", AtomSpecArg)],
        keyword=[("to", AtomSpecArg), ("MoveAtomSubset", BoolArg)],
        required_arguments=[],
        synopsis="Move a model or atoms to the center of another model without rotation.",
    )


__all__ = ["align_center", "align_center_desc", "define_centroid", "parse_map_or_atoms"]
