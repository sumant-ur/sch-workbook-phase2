"""Compatibility shim.

The admin panel UI was moved into `admin_config.py` so configuration persistence
and the UI live together.

You said you plan to remove this file later; until then, keep it so older
imports (`from admin_panel import display_super_admin_panel`) continue to work.
"""

from admin_config import display_super_admin_panel  # re-export  # noqa: F401
