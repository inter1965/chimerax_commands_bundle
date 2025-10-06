"""Reload scripts command."""

from __future__ import annotations


def reload_scripts(session):
    from chimerax.cmd_line.tool import CommandLine

    command_line = session.tools.find_by_class(CommandLine)[0]
    command_line._run_startup_commands()
    session.logger.status("Startup commands re-executed.", log=True)


def reload_scripts_desc():
    from chimerax.core.commands import CmdDesc

    return CmdDesc(
        required=[],
        keyword=[],
        required_arguments=[],
        synopsis="Reload startup commands.",
    )


__all__ = ["reload_scripts", "reload_scripts_desc"]
