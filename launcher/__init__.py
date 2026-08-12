"""Desktop launcher support for ProAim."""

from .settings import LauncherSettings, SettingsError, load_settings, save_settings

__all__ = [
    "LauncherSettings",
    "SettingsError",
    "load_settings",
    "save_settings",
]
