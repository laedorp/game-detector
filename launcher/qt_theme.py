"""Dark red visual theme for the Qt launcher.

The palette is deliberately conservative.  Pure black backgrounds with saturated
red text read as harsh and cheapen an interface, so the surfaces are very dark
neutrals with a slight blue cast and the red is reserved for a small number of
deliberate roles: the primary action, the active navigation item, focus rings,
and error states.  Everything else is carried by spacing and typography.
"""

from __future__ import annotations

from pathlib import Path
import sys


# Surfaces, from furthest back to closest to the reader.
BACKGROUND = "#0d0d10"
SURFACE = "#141418"
SURFACE_RAISED = "#1b1b21"
SURFACE_HOVER = "#22222a"
BORDER = "#2a2a33"
BORDER_STRONG = "#3a3a46"

# Red is an accent, not a background.  ACCENT is the resting state, ACCENT_HOVER
# lifts on hover, and ACCENT_DIM carries low-emphasis marks such as focus rings.
ACCENT = "#d21f3c"
ACCENT_HOVER = "#e63950"
ACCENT_PRESSED = "#a8172f"
ACCENT_DIM = "#5c1020"

TEXT = "#ecedef"
TEXT_MUTED = "#9a9ba4"
TEXT_FAINT = "#6b6c76"
TEXT_ON_ACCENT = "#ffffff"

SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER = "#f85149"

# An 8px rhythm keeps unrelated panels from drifting out of alignment.
UNIT = 8

def _assets_directory() -> Path:
    """Locate bundled assets in both a checkout and a frozen application.

    PyInstaller extracts data files to a temporary directory rather than beside
    the source, so resolving from ``__file__`` alone would find nothing once the
    application is packaged.
    """

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root) / "assets"
    return Path(__file__).resolve().parent.parent / "assets"


ASSETS = _assets_directory()


def stylesheet() -> str:
    """Return the application-wide Qt style sheet."""

    chevron = (ASSETS / "chevron-down.svg").as_posix()
    return f"""
    QWidget {{
        background-color: {BACKGROUND};
        color: {TEXT};
        font-family: "Inter", "Segoe UI", "Cantarell", "DejaVu Sans", sans-serif;
        font-size: 13px;
    }}

    /* Text and toggles must not paint their own rectangle.  A card sits on a
       lighter surface than the window, so an opaque child would stamp a
       visible band across it. */
    QLabel, QCheckBox, QRadioButton {{
        background-color: transparent;
    }}

    QLabel[role="title"] {{
        font-size: 19px;
        font-weight: 600;
        color: {TEXT};
    }}
    QLabel[role="subtitle"] {{
        font-size: 12px;
        color: {TEXT_MUTED};
    }}
    QLabel[role="sectionHeading"] {{
        font-size: 11px;
        font-weight: 700;
        color: {TEXT_FAINT};
        letter-spacing: 1px;
    }}
    QLabel[role="fieldLabel"] {{
        color: {TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel[role="status"] {{
        color: {TEXT_MUTED};
        font-size: 12px;
    }}
    QLabel[state="ok"] {{ color: {SUCCESS}; }}
    QLabel[state="warn"] {{ color: {WARNING}; }}
    QLabel[state="error"] {{ color: {DANGER}; }}

    /* Sidebar ---------------------------------------------------------- */
    QFrame#Sidebar {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    QPushButton[role="nav"] {{
        background-color: transparent;
        border: none;
        border-left: 3px solid transparent;
        padding: {UNIT + 3}px {UNIT + 4}px;
        text-align: left;
        color: {TEXT_MUTED};
        font-size: 13px;
        font-weight: 500;
    }}
    QPushButton[role="nav"]:hover {{
        background-color: {SURFACE_HOVER};
        color: {TEXT};
    }}
    QPushButton[role="nav"]:checked {{
        background-color: {SURFACE_RAISED};
        border-left: 3px solid {ACCENT};
        color: {TEXT};
        font-weight: 600;
    }}

    QFrame#Header {{
        background-color: {SURFACE};
        border-bottom: 1px solid {BORDER};
    }}
    QFrame#Footer {{
        background-color: {SURFACE};
        border-top: 1px solid {BORDER};
    }}

    /* Cards ------------------------------------------------------------ */
    QFrame[role="card"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 6px;
    }}

    /* Inputs ----------------------------------------------------------- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
        background-color: {SURFACE_RAISED};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 6px 8px;
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: {TEXT_ON_ACCENT};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {BORDER_STRONG};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT};
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled {{
        background-color: {SURFACE};
        color: {TEXT_FAINT};
        border-color: {BORDER};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox::down-arrow {{
        /* Qt stops drawing its native mark once the combo box is styled, so the
           chevron must be supplied as an image or the control loses its
           affordance entirely. */
        image: url("{chevron}");
        width: 12px;
        height: 12px;
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {SURFACE_RAISED};
        border: 1px solid {BORDER_STRONG};
        selection-background-color: {ACCENT};
        selection-color: {TEXT_ON_ACCENT};
        outline: none;
        padding: 2px;
    }}

    QSlider::groove:horizontal {{
        height: 6px;
        background: {BORDER};
        border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT_DIM};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        border: 1px solid {ACCENT_HOVER};
        width: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {ACCENT_HOVER};
    }}

    /* Buttons ---------------------------------------------------------- */
    QPushButton {{
        background-color: {SURFACE_RAISED};
        border: 1px solid {BORDER_STRONG};
        border-radius: 4px;
        padding: 7px 16px;
        color: {TEXT};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {SURFACE_HOVER};
        border-color: {TEXT_FAINT};
    }}
    QPushButton:pressed {{
        background-color: {SURFACE};
    }}
    QPushButton:disabled {{
        color: {TEXT_FAINT};
        border-color: {BORDER};
        background-color: {SURFACE};
    }}
    QPushButton[role="primary"] {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        color: {TEXT_ON_ACCENT};
        font-weight: 600;
        padding: 8px 22px;
    }}
    QPushButton[role="primary"]:hover {{
        background-color: {ACCENT_HOVER};
        border-color: {ACCENT_HOVER};
    }}
    QPushButton[role="primary"]:pressed {{
        background-color: {ACCENT_PRESSED};
    }}
    QPushButton[role="primary"]:disabled {{
        background-color: {ACCENT_DIM};
        border-color: {ACCENT_DIM};
        color: {TEXT_FAINT};
    }}

    /* Checkboxes and radios -------------------------------------------- */
    QCheckBox, QRadioButton {{
        spacing: 8px;
        color: {TEXT};
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        background-color: {SURFACE_RAISED};
        border: 1px solid {BORDER_STRONG};
    }}
    QCheckBox::indicator {{ border-radius: 3px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {ACCENT};
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}

    /* Log view --------------------------------------------------------- */
    QPlainTextEdit#LogView {{
        background-color: {BACKGROUND};
        border: 1px solid {BORDER};
        border-radius: 4px;
        font-family: "JetBrains Mono", "Cascadia Mono", "DejaVu Sans Mono", monospace;
        font-size: 12px;
        color: {TEXT_MUTED};
    }}

    /* Scrollbars ------------------------------------------------------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER_STRONG};
        border-radius: 5px;
        min-width: 28px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    QScrollArea {{ border: none; }}

    QToolTip {{
        background-color: {SURFACE_RAISED};
        color: {TEXT};
        border: 1px solid {BORDER_STRONG};
        padding: 5px 7px;
    }}

    QMessageBox {{ background-color: {SURFACE}; }}
    """
