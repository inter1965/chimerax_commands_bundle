"""Map eraser mask create command."""

from __future__ import annotations

import os


def _first_map(mask_arg):
    try:  # pragma: no cover - ChimeraX dependency
        from chimerax.core.models import Model
    except Exception:  # pragma: no cover - ChimeraX dependency
        Model = ()

    if isinstance(mask_arg, Model):
        first_map = mask_arg
    else:
        try:
            iterator = iter(mask_arg)
        except TypeError:
            first_map = mask_arg
        else:
            for first_map in iterator:
                break
            else:
                raise ValueError("No map supplied")

    if first_map is None:
        raise ValueError("No map supplied")
    return first_map


def map_eraser_mask_create(
    session,
    mask,
    sphere,
    save_masks=True,
    file_root=None,
    sphere_append="_sphere",
    full_append="_plus_sphere",
    extend=0.0,
    width=12,
):
    from chimerax.core.commands import CenterArg, run
    from chimerax.core.commands.cli import command_function
    from chimerax.map_eraser.eraser import volume_erase
    from chimerax.map_filter.vopcommand import volume_add, volume_threshold

    mask_volume = _first_map(mask)
    center = sphere.scene_position.origin()
    radius = sphere.radius
    center_arg = CenterArg().parse("%.5g,%.5g,%.5g" % tuple(center), session)[0]

    filled_v = volume_threshold(
        session,
        mask,
        minimum=2,
        set=1,
        maximum=2,
        set_maximum=0,
    )
    sphere_v = volume_erase(session, [filled_v], center=center_arg, radius=radius, outside=True)

    binary_v = volume_threshold(
        session,
        mask,
        minimum=0.5,
        set=0,
        maximum=0.5,
        set_maximum=1,
    )
    combined_v = volume_add(session, (sphere_v, binary_v))
    combined_v = volume_threshold(
        session,
        [combined_v],
        minimum=0.5,
        set=0,
        maximum=0.5,
        set_maximum=1,
    )

    soft_edge_mask = command_function("soft edge mask")
    soft_sphere = soft_edge_mask(session, sphere_v, level=0.5, extend=extend, width=width)
    soft_combined = soft_edge_mask(session, combined_v, level=0.5, extend=extend, width=width)

    if save_masks:
        if file_root is None:
            mask_path = getattr(mask_volume, "path", None)
            if mask_path:
                split_path = os.path.splitext(mask_path)
            else:
                split_path = ("mask", ".mrc")
        else:
            split_path = (file_root, "")

        sphere_path = split_path[0] + sphere_append + (split_path[1] or ".mrc")
        full_path = split_path[0] + full_append + (split_path[1] or ".mrc")
        run(session, f"save {sphere_path} format mrc models #{soft_sphere.id_string}")
        run(session, f"save {full_path} format mrc models #{soft_combined.id_string}")
        session.logger.status(f"Files output: {sphere_path} {full_path}", log=True)


def map_eraser_mask_create_desc():
    from chimerax.core.commands import BoolArg, CmdDesc, FloatArg, ModelArg, StringArg
    from chimerax.map import MapsArg

    return CmdDesc(
        required=[],
        keyword=[
            ("mask", MapsArg),
            ("sphere", ModelArg),
            ("save_masks", BoolArg),
            ("file_root", StringArg),
            ("sphere_append", StringArg),
            ("full_append", StringArg),
            ("extend", FloatArg),
            ("width", FloatArg),
        ],
        required_arguments=["mask", "sphere"],
        synopsis="Create a mask from the map eraser sphere.",
    )


__all__ = ["map_eraser_mask_create", "map_eraser_mask_create_desc"]
