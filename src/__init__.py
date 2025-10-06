"""ChimeraX custom functions bundle."""

from __future__ import annotations

from chimerax.core.toolshed import BundleAPI


class _CustomFunctionsAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from chimerax.core.commands import register

        from . import (
            align_center,
            align_symmetry_axis,
            fit_opposite_hand,
            map_eraser_mask_create,
            molmap_cube,
            reload_scripts,
            rough_fitmap,
            soft_edge_mask,
            to_residue,
        )

        command_map = {
            "align center": (align_center.align_center, align_center.align_center_desc),
            "align symmetry axis": (
                align_symmetry_axis.align_sym_axis,
                align_symmetry_axis.align_sym_axis_desc,
            ),
            "fit opposite hand": (
                fit_opposite_hand.fit_opposite_hand,
                fit_opposite_hand.fit_opposite_hand_desc,
            ),
            "first residue": (to_residue.first_residue, to_residue.first_residue_desc),
            "last residue": (to_residue.last_residue, to_residue.last_residue_desc),
            "map eraser mask create": (
                map_eraser_mask_create.map_eraser_mask_create,
                map_eraser_mask_create.map_eraser_mask_create_desc,
            ),
            "molmap cube": (molmap_cube.molmap_cube, molmap_cube.molmap_cube_desc),
            "next residue": (to_residue.next_residue, to_residue.next_residue_desc),
            "previous residue": (
                to_residue.previous_residue,
                to_residue.previous_residue_desc,
            ),
            "reload scripts": (reload_scripts.reload_scripts, reload_scripts.reload_scripts_desc),
            "rough fitmap": (rough_fitmap.rough_fitmap, rough_fitmap.rough_fitmap_desc),
            "soft edge mask": (
                soft_edge_mask.soft_edge_mask,
                soft_edge_mask.soft_edge_mask_desc,
            ),
            "to residue": (to_residue.to_residue, to_residue.to_residue_desc),
        }

        try:
            func, desc_func = command_map[ci.name]
        except KeyError as err:  # noqa: B904
            raise ValueError(f"trying to register unknown command: {ci.name}") from err

        desc = desc_func()
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, func, logger=logger)

    @staticmethod
    def initialize(session, bi):
        from . import to_residue

        to_residue.initialize(session)


bundle_api = _CustomFunctionsAPI()
