"""Small, native-widget dark theme for the desktop app (app spec §12)."""

from pathlib import Path

from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


COLORS = {
    "window": "#121214", "panel": "#1B1B1F", "field": "#222226",
    "hover": "#2A2A30", "pressed": "#151517", "selected": "#37373F",
    "viewer": "#0B0B0D", "divider": "#44444B", "control": "#707078",
    "text": "#EDEDED", "secondary": "#B5B5BD", "hint": "#93939F",
    "disabled": "#777780", "scarlet": "#E90802",
}


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    families = set(QFontDatabase.families())
    for family in ("Inter", "Helvetica Neue", "Tahoma", "Arial"):
        if family in families:
            font.setFamily(family)
            break
    # Keep the platform's point size and scaling rather than fixing physical pixels.
    app.setFont(font)
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "window", QPalette.ColorRole.WindowText: "text",
        QPalette.ColorRole.Base: "field", QPalette.ColorRole.AlternateBase: "panel",
        QPalette.ColorRole.Text: "text", QPalette.ColorRole.Button: "field",
        QPalette.ColorRole.ButtonText: "text", QPalette.ColorRole.Highlight: "selected",
        QPalette.ColorRole.HighlightedText: "text", QPalette.ColorRole.ToolTipBase: "field",
        QPalette.ColorRole.ToolTipText: "text", QPalette.ColorRole.PlaceholderText: "hint",
        QPalette.ColorRole.Link: "secondary", QPalette.ColorRole.Light: "control",
        QPalette.ColorRole.Mid: "divider", QPalette.ColorRole.Dark: "viewer",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(COLORS[token]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(COLORS["disabled"]))
    app.setPalette(palette)
    stylesheet = """
        QWidget { color: #EDEDED; }
        QMainWindow { background: #121214; }
        QWidget#inputRail, QDialog { background: #1B1B1F; }
        QWidget#productToolbar { background: #000000; }
        QLabel[role="muted"] { color: #B5B5BD; }
        QLabel[role="heading"] { font-weight: 600; }
        QLabel[role="error"] { color: #FFB4B0; background: #361B1D; padding: 8px; }
        QLabel[role="warning"] { color: #E9C46A; background: #30291C; padding: 8px; }
        QPushButton {
            background: #222226; border: 1px solid #707078; border-radius: 3px;
            padding: 6px 12px; min-height: 18px;
        }
        QPushButton:hover { background: #2A2A30; }
        QPushButton:pressed { background: #151517; }
        QPushButton:focus, QLineEdit:focus, QComboBox:focus {
            border: 2px solid #E90802;
        }
        QPushButton[primary="true"] { background: #E90802; color: #FFFFFF; font-weight: 600; }
        QPushButton[primary="true"]:hover { background: #C90702; }
        QPushButton[primary="true"]:pressed { background: #AA0602; }
        QPushButton[primary="true"]:focus { border: 2px solid #EDEDED; }
        QPushButton:disabled, QPushButton[primary="true"]:disabled {
            background: #222226; color: #777780; border: 1px solid #44444B;
        }
        /* Run/Cancel: 26 content + 12 padding + 2 border = 40 px (spec §12.4).
           The 2 px focus border takes 1 px of padding so the height stays fixed. */
        QPushButton#runButton { min-height: 26px; }
        QPushButton#runButton:focus { padding: 5px 11px; }
        QLineEdit, QComboBox {
            background: #222226; border: 1px solid #707078; border-radius: 3px;
            padding: 6px; min-height: 18px; selection-background-color: #37373F;
        }
        QComboBox QAbstractItemView {
            background: #222226; color: #EDEDED; selection-background-color: #37373F;
            selection-color: #EDEDED; border: 1px solid #707078;
        }
        QListView, QTableView {
            background: #1B1B1F; border: 1px solid #44444B;
            selection-background-color: #37373F; selection-color: #EDEDED;
        }
        QListView:focus, QTableView:focus { border: 2px solid #E90802; }
        QListView[dragOver="true"] { border: 2px solid #E90802; }
        QListView::item:selected:focus, QTableView::item:selected:focus {
            border: 1px solid #EDEDED;
        }
        QHeaderView::section {
            background: #222226; color: #EDEDED; padding: 6px;
            border: 0; border-bottom: 1px solid #44444B; font-weight: 600;
        }
        QTabWidget::pane { border: 0; background: #1B1B1F; }
        QTabBar::tab {
            background: #1B1B1F; color: #B5B5BD; padding: 8px 14px;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected { color: #EDEDED; border-bottom: 2px solid #E90802; }
        QTabBar::tab:focus { border: 1px solid #E90802; }
        QProgressBar {
            border: 1px solid #44444B; background: #222226; border-radius: 2px;
            min-height: 6px; max-height: 6px;
        }
        QProgressBar::chunk { background: #E90802; }
        QSplitter::handle { background: #44444B; }
        QToolTip { background: #222226; color: #EDEDED; border: 1px solid #707078; }
        QCheckBox, QRadioButton { spacing: 8px; padding: 3px 0; }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px; height: 16px; background: #222226; border: 1px solid #707078;
        }
        QCheckBox::indicator { border-radius: 2px; }
        QRadioButton::indicator { border-radius: 8px; }
        QCheckBox::indicator:checked { image: url("__CHECK_ICON__"); border: 1px solid transparent; }
        QRadioButton::indicator:checked { image: url("__RADIO_ICON__"); border: 1px solid transparent; }
        QCheckBox:focus, QRadioButton:focus { outline: 2px solid #E90802; }
        QStatusBar { background: #1B1B1F; border-top: 1px solid #44444B; }
    """
    assets = Path(__file__).resolve().parent / "assets"
    stylesheet = stylesheet.replace("__CHECK_ICON__", (assets / "check.svg").as_posix())
    stylesheet = stylesheet.replace("__RADIO_ICON__", (assets / "radio.svg").as_posix())
    app.setStyleSheet(stylesheet)
