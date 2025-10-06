"""Residue navigation commands."""

from __future__ import annotations

import numpy as np

from .align_center import define_centroid


def _resolve_selection(session, to_ends, first):
    from chimerax.atomic.structure import AtomicStructure

    residues_sets = session.selection.items("residues")
    if not residues_sets:
        session.logger.status(
            "First make an atomic selection (ctrl+click/drag or select command)",
            log=True,
        )
        return None

    atoms_sel = session.selection.items("atoms")
    structures = [m for m in session.selection.models() if isinstance(m, AtomicStructure)]
    if not structures:
        session.logger.status("No atomic models selected.", log=True)
        return None

    index = 0 if first else -1
    residues_set = residues_sets[index]
    if not residues_set:
        session.logger.status("Selection does not contain residues.", log=True)
        return None
    residue = residues_set.filter([0 if first else len(residues_set) - 1])[0]
    structure = structures[index]

    selection_same = (
        len(atoms_sel) == 1 and np.array_equal(atoms_sel[0].pointers, residue.atoms.pointers)
    )

    if selection_same and to_ends:
        target_index = 0 if first else len(structure.residues) - 1
    else:
        target_index = structure.residues.index(residue)

    return structure, target_index, selection_same


def _select_residue(session, residue, move=True):
    from chimerax.core.commands import run

    atomspec = residue.atomspec
    model = residue.structure.id_string
    if not atomspec.startswith("#"):
        spec = f"#{model}{atomspec}"
    else:
        spec = atomspec

    run(session, f"select {spec}")

    name = residue.name
    code = residue.one_letter_code or ""
    message = f"{spec} {name} {code}"

    if move:
        center = define_centroid(session, residue.atoms)
        s2c = session.main_view.camera.position.inverse()
        residue_center = s2c.transform_points(np.expand_dims(center, 0))
        current_center = s2c.transform_points(np.expand_dims(session.main_view.center_of_rotation, 0))
        center_dif = [-1 * (m - a) for m, a in zip(residue_center[0], current_center[0])]
        run(session, "move %0.2f,%0.2f,%0.2f" % (center_dif[0], center_dif[1], center_dif[2]))
        run(session, "cofr sel")

    session.logger.status(message, log=True)


def go_to_residue(session, step, to_ends=False, first=True, NoMove=False):
    resolved = _resolve_selection(session, to_ends, first)
    if resolved is None:
        return False
    structure, index, selection_same = resolved

    if selection_same:
        new_index = index + step
    else:
        new_index = index

    if new_index < 0:
        session.logger.status("Start of chain.", log=True)
        new_index = 0
    elif new_index > len(structure.residues) - 1:
        session.logger.status("End of chain.", log=True)
        new_index = len(structure.residues) - 1

    residue = structure.residues[new_index]
    _select_residue(session, residue, move=not NoMove)
    return True


def next_residue(session, NoMove=False):
    go_to_residue(session, 1, NoMove=NoMove)


def previous_residue(session, NoMove=False):
    go_to_residue(session, -1, NoMove=NoMove)


def first_residue(session, NoMove=False):
    go_to_residue(session, 0, to_ends=True, NoMove=NoMove)


def last_residue(session, NoMove=False):
    go_to_residue(session, 0, to_ends=True, first=False, NoMove=NoMove)


def to_residue(session, NoMove=False):
    go_to_residue(session, 0, NoMove=NoMove)


def _residue_desc(synopsis):
    from chimerax.core.commands import BoolArg, CmdDesc

    return CmdDesc(
        required=[],
        keyword=[("NoMove", BoolArg)],
        required_arguments=[],
        synopsis=synopsis,
    )


def next_residue_desc():
    return _residue_desc("Go to next residue.")


def previous_residue_desc():
    return _residue_desc("Go to previous residue.")


def first_residue_desc():
    return _residue_desc("Go to first residue of selection.")


def last_residue_desc():
    return _residue_desc("Go to last residue of selection.")


def to_residue_desc():
    return _residue_desc("Identify current residue of selection.")


def create_button_panel(session):
    from chimerax.buttonpanel import buttons
    from chimerax.core.commands import run

    button_panel_title = "Residue shortcuts"
    bp = buttons._button_panel_with_title(session, button_panel_title)
    if bp:
        bp.tool_window.destroy()
        bp.tool_window.cleanup()
        session._button_panels = [b for b in buttons._button_panels(session) if b is not bp]

    run(session, f"buttonpanel \"{button_panel_title}\" rows 2 columns 3")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"To residue\" command \"to residue\"")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"Previous residue\" command \"previous residue\"")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"Next residue\" command \"next residue\"")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"Reset cofr\" command \"cofr all\"")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"First residue\" command \"first residue\"")
    run(session, f"buttonpanel \"{button_panel_title}\" add \"Last residue\" command \"last residue\"")


def initialize(session):
    try:
        create_button_panel(session)
    except Exception as err:  # noqa: BLE001
        session.logger.warning(f"Unable to create residue shortcut panel: {err}")


__all__ = [
    "create_button_panel",
    "first_residue",
    "first_residue_desc",
    "initialize",
    "last_residue",
    "last_residue_desc",
    "next_residue",
    "next_residue_desc",
    "previous_residue",
    "previous_residue_desc",
    "to_residue",
    "to_residue_desc",
]
