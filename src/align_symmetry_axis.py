"""Align symmetry axis command implementation."""

from __future__ import annotations

import numpy as np

from .align_center import define_centroid


def is_planar(points, tolerance=1):
    points = np.asarray(points)
    if len(points) < 4:
        return True

    p0, p1, p2 = points[:3]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return False
    normal /= norm

    for pi in points[3:]:
        distance = np.dot(pi - p0, normal)
        if abs(distance) > tolerance:
            return False
    return True


def align_sym_axis(session, atoms, sym, MoveToOrigin=True):
    from chimerax.core.commands import run
    from chimerax.core.errors import UserError
    from chimerax.geometry import rotation

    if not sym.lower().startswith("c"):
        raise UserError(f"Cyclic symmetry only accepted. Not {sym}.")

    try:
        cyclic_sym = int(sym[1:])
    except Exception as err:  # noqa: BLE001
        raise UserError(f"Cannot parse symmetry from {sym}. Should be C2 or C3 etc.") from err

    if cyclic_sym < 3:
        raise UserError("Symmetries less than C3 currently not supported.")

    num_atoms = len(atoms)
    if num_atoms != cyclic_sym:
        raise UserError(f"Number of atoms ({num_atoms}) must match symmetry order (C{cyclic_sym}).")

    scene_coords = atoms.scene_coords
    if not is_planar(scene_coords):
        raise UserError("For symmetries greater than C3, supplied points must be co-planar.")

    centroid = define_centroid(session, atoms)

    p0, p1, p2 = scene_coords[:3]
    v0 = p1 - p0
    v1 = p2 - p0
    normal = np.cross(v0, v1)
    normal /= np.linalg.norm(normal)
    target = np.array([0, 0, 1])
    axis = np.cross(normal, target)
    angle = np.degrees(np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0)))
    transform = rotation(axis, angle, centroid)

    run(session, "view orient")
    atoms[0].structure.atoms.transform(transform)

    if MoveToOrigin:
        model_id = atoms[0].structure.atomspec
        s2c = session.main_view.camera.position.inverse()
        screen_atoms_center = s2c.transform_points(np.expand_dims(centroid, 0))
        screen_origin_center = s2c.transform_points(np.expand_dims([0.0, 0.0, 0.0], 0))
        center_dif = [-1 * (m - a) for m, a in zip(screen_atoms_center[0], screen_origin_center[0])]
        run(
            session,
            "move %0.2f,%0.2f,%0.2f model %s"
            % (center_dif[0], center_dif[1], center_dif[2], model_id),
        )
        run(session, "view orient")
        msg_str = f"Symmetry axis of model {model_id} aligned to Z axis and centroid moved to origin."
    else:
        msg_str = "Symmetry axis aligned to Z axis."

    session.logger.status(msg_str, log=True)


def align_sym_axis_desc():
    from chimerax.atomic import AtomsArg
    from chimerax.core.commands import BoolArg, CmdDesc, StringArg

    return CmdDesc(
        required=[("atoms", AtomsArg), ("sym", StringArg)],
        keyword=[("MoveToOrigin", BoolArg)],
        required_arguments=[],
        synopsis="Align a symmetric model to the z axis. Cyclic symmetry only.",
    )


__all__ = ["align_sym_axis", "align_sym_axis_desc"]
