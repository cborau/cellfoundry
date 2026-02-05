import os
import re
import sys
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

from PySide6.QtCore import Qt, QTimer, QProcess, QEvent, QRegularExpression
from PySide6.QtGui import QColor, QFont, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QIcon, QPalette
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QPlainTextEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton, QLineEdit,
    QCheckBox, QLabel, QMessageBox, QSplitter, QGridLayout
)


# ----------------------------
# Minimal Python highlighter
# ----------------------------
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)

        def fmt(color: str, bold: bool = False, italic: bool = False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Bold)
            if italic:
                f.setFontItalic(True)
            return f

        self.f_comment = fmt("#6A9955", italic=True)
        self.f_string = fmt("#CE9178")
        self.f_number = fmt("#B5CEA8")
        self.f_keyword = fmt("#C586C0", bold=True)
        self.f_builtin = fmt("#4EC9B0")
        self.f_caps = fmt("#4FC1FF", bold=True)
        self.f_function = fmt("#DCDCAA")
        self.f_class = fmt("#4EC9B0", bold=True)
        self.f_operator = fmt("#D4D4D4")
        self.f_bracket = fmt("#D4D4D4")
        self.section_bg = QColor("#E09470")
        self.f_section_bg = QTextCharFormat()
        self.f_section_bg.setBackground(self.section_bg)
        self.section_fg = QColor("#2B1D0F")
        self.f_section_line = QTextCharFormat()
        self.f_section_line.setBackground(self.section_bg)
        self.f_section_line.setForeground(self.section_fg)
        self.subsection_bg = QColor("#C7AA7F")
        self.f_subsection_bg = QTextCharFormat()
        self.f_subsection_bg.setBackground(self.subsection_bg)
        self.subsection_fg = QColor("#3A2C14")
        self.f_subsection_line = QTextCharFormat()
        self.f_subsection_line.setBackground(self.subsection_bg)
        self.f_subsection_line.setForeground(self.subsection_fg)

        self.re_comment = re.compile(r"#.*$")
        self.re_string = re.compile(r"(\"\"\".*?\"\"\"|'''.*?'''|\".*?\"|'.*?')")  # simple
        self.re_number = re.compile(r"\b(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\b")

        kws = [
            "False", "True", "None", "and", "or", "not", "if", "elif", "else", "for", "while",
            "break", "continue", "return", "def", "class", "import", "from", "as", "try", "except",
            "finally", "with", "pass", "raise", "in", "is", "lambda", "yield", "global", "nonlocal",
        ]
        self.re_keyword = re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b")

        builtins_ = ["int", "float", "str", "list", "dict", "set", "tuple", "len", "range", "print"]
        self.re_builtin = re.compile(r"\b(" + "|".join(map(re.escape, builtins_)) + r")\b")

        self.re_caps = re.compile(r"\b[A-Z_]{3,}\b")
        self.re_function = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)")
        self.re_class = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)")
        self.re_operator = re.compile(r"(//|<<|>>|==|!=|<=|>=|\+|\-|\*|/|%|=|<|>|&|\||\^|~)")
        self.re_bracket = re.compile(r"[\(\)\[\]\{\}]")

    def highlightBlock(self, text: str):
        is_sub_border = re.match(r"^\s*# \+\+[=\-]+\+\+\s*$", text) is not None
        is_sub_title = re.match(r"^\s*# \+\+\s*.+$", text) is not None
        is_section_border = re.match(r"^\s*# \+[=\-]+\+\s*$", text) is not None
        is_section_title = re.match(r"^\s*# \|.*\|\s*$", text) is not None

        in_sub_line = is_sub_border or is_sub_title
        in_section_line = (is_section_border or is_section_title) and not in_sub_line

        if in_sub_line:
            self.setFormat(0, len(text), self.f_subsection_line)
            self.setCurrentBlockState(0)
            return
        elif in_section_line:
            self.setFormat(0, len(text), self.f_section_line)
            self.setCurrentBlockState(0)
            return

        def set_fmt(start: int, length: int, fmt: QTextCharFormat):
            if in_sub_line:
                f = QTextCharFormat(fmt)
                f.setBackground(self.subsection_bg)
                self.setFormat(start, length, f)
            elif in_section_line:
                f = QTextCharFormat(fmt)
                f.setBackground(self.section_bg)
                self.setFormat(start, length, f)
            else:
                self.setFormat(start, length, fmt)

        # Comments
        m = self.re_comment.search(text)
        if m:
            start, end = m.start(), m.end()
            set_fmt(start, end - start, self.f_comment)
            code_part = text[:start]
        else:
            code_part = text

        # Strings
        for m in self.re_string.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_string)

        # Numbers
        for m in self.re_number.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_number)

        # Keywords and builtins
        for m in self.re_keyword.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_keyword)
        for m in self.re_builtin.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_builtin)

        # ALL_CAPS variables
        for m in self.re_caps.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_caps)

        # Function and class names
        for m in self.re_function.finditer(code_part):
            set_fmt(m.start(1), m.end(1) - m.start(1), self.f_function)
        for m in self.re_class.finditer(code_part):
            set_fmt(m.start(1), m.end(1) - m.start(1), self.f_class)

        # Operators and brackets
        for m in self.re_operator.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_operator)
        for m in self.re_bracket.finditer(code_part):
            set_fmt(m.start(), m.end() - m.start(), self.f_bracket)

        self.setCurrentBlockState(0)


# ----------------------------
# File patching utilities
# ----------------------------
_ASSIGN_RE = re.compile(r"^(\s*)([A-Z_][A-Z0-9_]*)\s*=\s*(.*)$")

def split_rhs_and_comment(rhs: str) -> Tuple[str, str]:
    """
    Split "value  # comment" -> ("value", "  # comment")
    Simple heuristic: first # not in quotes.
    """
    in_s = False
    in_d = False
    esc = False
    for i, ch in enumerate(rhs):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return rhs[:i].rstrip(), rhs[i:]
    return rhs.rstrip(), ""


def py_bool(v: bool) -> str:
    return "True" if v else "False"


def py_num_text(s: str) -> str:
    # Keep user typing flexible, but validate later if you want stricter.
    return s.strip()


def parse_csv_numbers(text: str) -> List[float]:
    # accepts "1, 2, 3.5" and also spaces/newlines
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    parts = [p for p in parts if p != ""]
    out = []
    for p in parts:
        out.append(float(p))
    return out


def parse_csv_int01(text: str) -> List[int]:
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    parts = [p for p in parts if p != ""]
    out = []
    for p in parts:
        iv = int(p)
        if iv not in (0, 1):
            raise ValueError("Only 0/1 allowed")
        out.append(iv)
    return out


def format_list(vals: List[float], as_int: bool = False) -> str:
    if as_int:
        inner = ", ".join(str(int(v)) for v in vals)
    else:
        # repr(float) is fine; if you want consistent formatting, change here
        inner = ", ".join(repr(float(v)) for v in vals)
    return "[" + inner + "]"


@dataclass
class VarLoc:
    line_index: int
    indent: str
    var: str
    comment: str


def build_var_index(lines: List[str], vars_needed: List[str]) -> Dict[str, VarLoc]:
    locs: Dict[str, VarLoc] = {}
    needed = set(vars_needed)
    for i, line in enumerate(lines):
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        indent, var, rhs = m.group(1), m.group(2), m.group(3)
        if var in needed and var not in locs:
            rhs_val, rhs_comment = split_rhs_and_comment(rhs)
            locs[var] = VarLoc(line_index=i, indent=indent, var=var, comment=rhs_comment.rstrip("\n"))
    return locs


def replace_var_line(lines: List[str], locs: Dict[str, VarLoc], var: str, new_rhs: str) -> None:
    if var not in locs:
        raise KeyError(f"Variable {var} not found in file.")
    loc = locs[var]
    comment = loc.comment
    # preserve newline ending
    nl = "\n" if lines[loc.line_index].endswith("\n") else ""
    comment_part = f" {comment}" if comment and not comment.startswith("#") else (comment if comment else "")
    lines[loc.line_index] = f"{loc.indent}{var} = {new_rhs}{comment_part}{nl}"


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Parameter Editor")

        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.path: Optional[str] = None
        self.lines: List[str] = []
        self.locs: Dict[str, VarLoc] = {}

        self.process: Optional[QProcess] = None
        self._suppress_editor_signal = False
        self._editor_dirty = False
        self.vars_needed = [
            "TIME_STEP", "STEPS", "SAVE_EVERY_N_STEPS",
            "BOUNDARY_COORDS", "BOUNDARY_DISP_RATES", "BOUNDARY_DISP_RATES_PARALLEL",
            "CLAMP_AGENT_TOUCHING_BOUNDARY", "ALLOW_AGENT_SLIDING",
            "INCLUDE_FIBRE_NETWORK", "INCLUDE_DIFFUSION", "INCLUDE_CELLS"
        ]

        root = QWidget()
        self.setCentralWidget(root)

        # Editor
        self.editor = QPlainTextEdit()
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setFont(QFont("Consolas", 10))
        self.highlighter = PythonHighlighter(self.editor.document())
        self.bg_color = "#1E1E1E"
        self.fg_color = "#D4D4D4"
        self.selection_color = "#264F78"
        self.current_line_color = "#2A2D2E"

        editor_palette = self.editor.palette()
        editor_palette.setColor(QPalette.Base, QColor(self.bg_color))
        editor_palette.setColor(QPalette.Text, QColor(self.fg_color))
        editor_palette.setColor(QPalette.Highlight, QColor(self.selection_color))
        editor_palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        self.editor.setPalette(editor_palette)
        self.editor.cursorPositionChanged.connect(self._highlight_current_line)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        log_palette = self.log.palette()
        log_palette.setColor(QPalette.Base, QColor(self.bg_color))
        log_palette.setColor(QPalette.Text, QColor(self.fg_color))
        log_palette.setColor(QPalette.Highlight, QColor(self.selection_color))
        log_palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        self.log.setPalette(log_palette)

        # Controls
        top = QHBoxLayout()
        btn_open = QPushButton("Open")
        btn_save = QPushButton("Save")
        btn_reload = QPushButton("Reload")
        btn_run = QPushButton("Run")
        btn_stop = QPushButton("Stop")
        btn_stop.setEnabled(False)
        btn_toggle_log = QPushButton("Log only")

        top.addWidget(btn_open)
        top.addWidget(btn_save)
        top.addWidget(btn_reload)
        top.addStretch(1)
        top.addWidget(btn_run)
        top.addWidget(btn_stop)
        top.addWidget(btn_toggle_log)

        btn_open.clicked.connect(self.on_open)
        btn_save.clicked.connect(self.on_save)
        btn_reload.clicked.connect(self.on_reload)
        btn_run.clicked.connect(self.on_run)
        btn_stop.clicked.connect(self.on_stop)
        btn_toggle_log.clicked.connect(self.toggle_log_view)

        self.btn_save = btn_save
        self.btn_run = btn_run
        self.btn_stop = btn_stop
        self.btn_toggle_log = btn_toggle_log
        self._log_only = False
        self._splitter_v_sizes = None

        # Parameter panels
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # 1) GLOBAL PARAMETERS
        gb_global = QGroupBox("GLOBAL PARAMETERS")
        fl_global = QFormLayout(gb_global)
        self.ed_time_step = QLineEdit()
        self.ed_steps = QLineEdit()
        self.ed_save_every = QLineEdit()
        fl_global.addRow("TIME_STEP", self.ed_time_step)
        fl_global.addRow("STEPS", self.ed_steps)
        fl_global.addRow("SAVE_EVERY_N_STEPS", self.ed_save_every)
        self.btn_goto_global = QPushButton("Go to section")
        goto_global_row = QHBoxLayout()
        goto_global_row.addStretch(1)
        goto_global_row.addWidget(self.btn_goto_global)
        fl_global.addRow("", goto_global_row)

        # 2) BOUNDARY BEHAVIOUR
        gb_boundary = QGroupBox("BOUNDARY BEHAVIOUR")
        fl_boundary = QFormLayout(gb_boundary)
        self.ed_boundary_coords = [QLineEdit() for _ in range(6)]
        self.ed_boundary_disp = [QLineEdit() for _ in range(6)]
        self.ed_boundary_disp_par = [QLineEdit() for _ in range(12)]
        axis_labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
        self.cb_clamp = [QCheckBox(lbl) for lbl in axis_labels]
        self.cb_slide = [QCheckBox(lbl) for lbl in axis_labels]
        coords_widget = QWidget()
        coords_layout = QHBoxLayout(coords_widget)
        coords_layout.setContentsMargins(0, 0, 0, 0)
        for lbl, ed in zip(axis_labels, self.ed_boundary_coords):
            coords_layout.addWidget(QLabel(lbl))
            coords_layout.addWidget(ed)

        disp_widget = QWidget()
        disp_layout = QHBoxLayout(disp_widget)
        disp_layout.setContentsMargins(0, 0, 0, 0)
        for lbl, ed in zip(axis_labels, self.ed_boundary_disp):
            disp_layout.addWidget(QLabel(lbl))
            disp_layout.addWidget(ed)

        disp_par_labels = ["+x_y", "+x_z", "-x_y", "-x_z", "+y_x", "+y_z", "-y_x", "-y_z", "+z_x", "+z_y", "-z_x", "-z_y"]
        disp_par_widget = QWidget()
        disp_par_layout = QVBoxLayout(disp_par_widget)
        disp_par_layout.setContentsMargins(0, 0, 0, 0)
        for row in range(3):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for col in range(4):
                idx = row * 4 + col
                row_layout.addWidget(QLabel(disp_par_labels[idx]))
                row_layout.addWidget(self.ed_boundary_disp_par[idx])
            row_layout.addStretch(1)
            disp_par_layout.addWidget(row_widget)

        fl_boundary.addRow("BOUNDARY_COORDS", coords_widget)
        fl_boundary.addRow("BOUNDARY_DISP_RATES", disp_widget)
        fl_boundary.addRow("BOUNDARY_DISP_RATES_PARALLEL", disp_par_widget)
        clamp_widget = QWidget()
        clamp_layout = QHBoxLayout(clamp_widget)
        clamp_layout.setContentsMargins(0, 0, 0, 0)
        for cb in self.cb_clamp:
            clamp_layout.addWidget(cb)
        clamp_layout.addStretch(1)

        slide_widget = QWidget()
        slide_layout = QHBoxLayout(slide_widget)
        slide_layout.setContentsMargins(0, 0, 0, 0)
        for cb in self.cb_slide:
            slide_layout.addWidget(cb)
        slide_layout.addStretch(1)

        fl_boundary.addRow("CLAMP_AGENT_TOUCHING_BOUNDARY", clamp_widget)
        fl_boundary.addRow("ALLOW_AGENT_SLIDING", slide_widget)
        self.btn_goto_boundary = QPushButton("Go to section")
        goto_boundary_row = QHBoxLayout()
        goto_boundary_row.addStretch(1)
        goto_boundary_row.addWidget(self.btn_goto_boundary)
        fl_boundary.addRow("", goto_boundary_row)

        # 3) FIBRE NETWORK
        gb_fibre = QGroupBox("FIBRE NETWORK")
        vb_fibre = QVBoxLayout(gb_fibre)
        self.cb_fibre = QCheckBox("INCLUDE_FIBRE_NETWORK")
        self.btn_goto_fibre = QPushButton("Go to section")
        row_fibre = QHBoxLayout()
        row_fibre.addWidget(self.cb_fibre)
        row_fibre.addStretch(1)
        row_fibre.addWidget(self.btn_goto_fibre)
        vb_fibre.addLayout(row_fibre)

        # 4) SPECIES DIFFUSION
        gb_diff = QGroupBox("SPECIES DIFFUSION")
        vb_diff = QVBoxLayout(gb_diff)
        self.cb_diff = QCheckBox("INCLUDE_DIFFUSION")
        self.btn_goto_diff = QPushButton("Go to section")
        row_diff = QHBoxLayout()
        row_diff.addWidget(self.cb_diff)
        row_diff.addStretch(1)
        row_diff.addWidget(self.btn_goto_diff)
        vb_diff.addLayout(row_diff)

        # 5) CELLS
        gb_cells = QGroupBox("CELLS")
        vb_cells = QVBoxLayout(gb_cells)
        self.cb_cells = QCheckBox("INCLUDE_CELLS")
        self.btn_goto_cells = QPushButton("Go to section")
        row_cells = QHBoxLayout()
        row_cells.addWidget(self.cb_cells)
        row_cells.addStretch(1)
        row_cells.addWidget(self.btn_goto_cells)
        vb_cells.addLayout(row_cells)

        # 6) FLAMEGPU IMPLEMENTATION
        gb_impl = QGroupBox("FLAMEGPU IMPLEMENTATION")
        grid_impl = QGridLayout(gb_impl)

        def mk_impl_item(label_text: str):
            label = QLabel(label_text)
            btn = QPushButton("Go to section")
            row = QHBoxLayout()
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(btn)
            row_widget = QWidget()
            row_widget.setLayout(row)
            return btn, row_widget

        self.btn_impl_files, w_files = mk_impl_item("Files")
        self.btn_impl_globals, w_globals = mk_impl_item("Globals")
        self.btn_impl_messages, w_messages = mk_impl_item("Messages")
        self.btn_impl_agents, w_agents = mk_impl_item("Agents")
        self.btn_impl_step_funcs, w_step_funcs = mk_impl_item("Step functions")
        self.btn_impl_layers, w_layers = mk_impl_item("Layers")
        self.btn_impl_logging, w_logging = mk_impl_item("Logging")
        self.btn_impl_visualization, w_visualization = mk_impl_item("Visualization")
        self.btn_impl_execution, w_execution = mk_impl_item("Execution")

        grid_impl.addWidget(w_files, 0, 0)
        grid_impl.addWidget(w_globals, 0, 1)
        grid_impl.addWidget(w_messages, 0, 2)
        grid_impl.addWidget(w_agents, 1, 0)
        grid_impl.addWidget(w_step_funcs, 1, 1)
        grid_impl.addWidget(w_layers, 1, 2)
        grid_impl.addWidget(w_logging, 2, 0)
        grid_impl.addWidget(w_visualization, 2, 1)
        grid_impl.addWidget(w_execution, 2, 2)

        gb_title_font = QFont()
        gb_title_font.setBold(True)
        gb_content_font = QFont()
        gb_content_font.setBold(False)
        for gb in [gb_global, gb_boundary, gb_fibre, gb_diff, gb_cells, gb_impl]:
            gb.setFont(gb_title_font)
            for child in gb.findChildren(QWidget):
                child.setFont(gb_content_font)

        panel_layout.addWidget(gb_global)
        panel_layout.addWidget(gb_boundary)
        panel_layout.addWidget(gb_fibre)
        panel_layout.addWidget(gb_diff)
        panel_layout.addWidget(gb_cells)
        panel_layout.addWidget(gb_impl)
        panel_layout.addStretch(1)

        # Split view: left parameters, right editor, bottom log
        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.addWidget(panel)
        splitter_h.addWidget(self.editor)
        splitter_h.setStretchFactor(0, 0)
        splitter_h.setStretchFactor(1, 1)
        splitter_h.setSizes([300, 900])

        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(splitter_h)
        splitter_v.addWidget(self.log)
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 1)

        self.splitter_h = splitter_h
        self.splitter_v = splitter_v

        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(splitter_v)

        # Wire auto-update
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.apply_all_fields_to_editor)

        self._editor_debounce = QTimer()
        self._editor_debounce.setSingleShot(True)
        self._editor_debounce.timeout.connect(self._refresh_from_editor)

        for w in [
            self.ed_time_step, self.ed_steps, self.ed_save_every,
            *self.ed_boundary_coords, *self.ed_boundary_disp, *self.ed_boundary_disp_par
        ]:
            w.textChanged.connect(self._debounced_apply)
            w.installEventFilter(self)

        for cb in self.cb_clamp:
            cb.stateChanged.connect(lambda _, v="CLAMP_AGENT_TOUCHING_BOUNDARY", boxes=self.cb_clamp: self.on_toggle_list(v, boxes, goto=True))
        for cb in self.cb_slide:
            cb.stateChanged.connect(lambda _, v="ALLOW_AGENT_SLIDING", boxes=self.cb_slide: self.on_toggle_list(v, boxes, goto=True))

        self.cb_fibre.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_FIBRE_NETWORK", self.cb_fibre, goto=True))
        self.cb_diff.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_DIFFUSION", self.cb_diff, goto=True))
        self.cb_cells.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_CELLS", self.cb_cells, goto=True))

        self.btn_goto_global.clicked.connect(lambda: self.goto_var("TIME_STEP"))
        self.btn_goto_boundary.clicked.connect(lambda: self.goto_var("BOUNDARY_COORDS"))
        self.btn_goto_fibre.clicked.connect(lambda: self.goto_var("INCLUDE_FIBRE_NETWORK"))
        self.btn_goto_diff.clicked.connect(lambda: self.goto_var("INCLUDE_DIFFUSION"))
        self.btn_goto_cells.clicked.connect(lambda: self.goto_var("INCLUDE_CELLS"))
        self.btn_impl_files.clicked.connect(lambda: self.goto_subsection("Files"))
        self.btn_impl_globals.clicked.connect(lambda: self.goto_subsection("Globals"))
        self.btn_impl_messages.clicked.connect(lambda: self.goto_subsection("Messages"))
        self.btn_impl_agents.clicked.connect(lambda: self.goto_subsection("Agents"))
        self.btn_impl_step_funcs.clicked.connect(lambda: self.goto_subsection("Step functions"))
        self.btn_impl_layers.clicked.connect(lambda: self.goto_subsection("Layers"))
        self.btn_impl_logging.clicked.connect(lambda: self.goto_subsection("Logging"))
        self.btn_impl_visualization.clicked.connect(lambda: self.goto_subsection("Visualization"))
        self.btn_impl_execution.clicked.connect(lambda: self.goto_subsection("Execution"))

        self.editor.textChanged.connect(self._on_editor_text_changed)

        self.set_ui_enabled(False)

    def set_ui_enabled(self, enabled: bool):
        for w in [
            self.ed_time_step, self.ed_steps, self.ed_save_every,
            *self.ed_boundary_coords, *self.ed_boundary_disp, *self.ed_boundary_disp_par,
            self.cb_fibre, self.cb_diff, self.cb_cells,
            self.btn_goto_global, self.btn_goto_boundary,
            self.btn_goto_fibre, self.btn_goto_diff, self.btn_goto_cells,
            self.btn_impl_files, self.btn_impl_globals, self.btn_impl_messages,
            self.btn_impl_agents, self.btn_impl_step_funcs, self.btn_impl_layers,
            self.btn_impl_logging, self.btn_impl_visualization, self.btn_impl_execution,
            self.btn_save, self.btn_run
        ]:
            w.setEnabled(enabled)

    def _debounced_apply(self):
        # debounce to avoid rewriting the editor on every keystroke
        self._debounce.start(150)

    def toggle_log_view(self):
        if not self._log_only:
            self._splitter_v_sizes = self.splitter_v.sizes()
            self.splitter_v.setSizes([0, 1])
            self.btn_toggle_log.setText("Show editor")
            self._log_only = True
        else:
            if self._splitter_v_sizes:
                self.splitter_v.setSizes(self._splitter_v_sizes)
            else:
                self.splitter_v.setSizes([1, 1])
            self.btn_toggle_log.setText("Log only")
            self._log_only = False

    def _highlight_current_line(self):
        if self.editor.isReadOnly():
            return
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor(self.current_line_color))
        selection.format.setProperty(QTextCharFormat.FullWidthSelection, True)
        selection.cursor = self.editor.textCursor()
        selection.cursor.clearSelection()
        self.editor.setExtraSelections([selection])

    def _prepare_for_goto(self):
        # Ensure pending UI/editor updates are applied before navigation
        if self._debounce.isActive():
            self._debounce.stop()
            self.apply_all_fields_to_editor()
        if self._editor_debounce.isActive():
            self._editor_debounce.stop()
            self._refresh_from_editor()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open simulation file", "", "Python files (*.py);;All files (*)")
        if not path:
            return
        self.load_file(path)

    def on_reload(self):
        if not self.path:
            return
        self.load_file(self.path)

    def load_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return

        self.path = path
        self.lines = text.splitlines(True)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False

        self.locs = build_var_index(self.lines, self.vars_needed)

        missing = [v for v in self.vars_needed if v not in self.locs]
        if missing:
            self.log.append(f"Warning: could not find variables: {missing}")

        # Populate UI from file (best-effort text extraction)
        def get_rhs_text(var: str) -> Optional[str]:
            if var not in self.locs:
                return None
            li = self.locs[var].line_index
            m = _ASSIGN_RE.match(self.lines[li])
            if not m:
                return None
            rhs = m.group(3)
            rhs_val, _ = split_rhs_and_comment(rhs)
            return rhs_val.strip()

        self.ed_time_step.setText(get_rhs_text("TIME_STEP") or "")
        self.ed_steps.setText(get_rhs_text("STEPS") or "")
        self.ed_save_every.setText(get_rhs_text("SAVE_EVERY_N_STEPS") or "")

        # lists shown as csv in UI
        def list_to_csv(rhs: Optional[str]) -> str:
            if not rhs:
                return ""
            # strip [] or () if present
            t = rhs.strip()
            if (t.startswith("[") and t.endswith("]")) or (t.startswith("(") and t.endswith(")")):
                t = t[1:-1]
            return t.strip()

        def rhs_to_float_list(rhs: Optional[str], n: int) -> List[float]:
            if not rhs:
                return [0.0] * n
            try:
                t = list_to_csv(rhs)
                vals = parse_csv_numbers(t)
                if len(vals) < n:
                    vals = vals + [0.0] * (n - len(vals))
                return vals[:n]
            except Exception:
                return [0.0] * n

        coords_vals = rhs_to_float_list(get_rhs_text("BOUNDARY_COORDS"), 6)
        disp_vals = rhs_to_float_list(get_rhs_text("BOUNDARY_DISP_RATES"), 6)
        disp_par_vals = rhs_to_float_list(get_rhs_text("BOUNDARY_DISP_RATES_PARALLEL"), 12)

        for i, ed in enumerate(self.ed_boundary_coords):
            ed.setText(str(coords_vals[i]))
        for i, ed in enumerate(self.ed_boundary_disp):
            ed.setText(str(disp_vals[i]))
        for i, ed in enumerate(self.ed_boundary_disp_par):
            ed.setText(str(disp_par_vals[i]))
        def rhs_to_int_list(rhs: Optional[str]) -> List[int]:
            if not rhs:
                return [0] * 6
            try:
                t = list_to_csv(rhs)
                vals = parse_csv_int01(t)
                if len(vals) < 6:
                    vals = vals + [0] * (6 - len(vals))
                return vals[:6]
            except Exception:
                return [0] * 6

        clamp_vals = rhs_to_int_list(get_rhs_text("CLAMP_AGENT_TOUCHING_BOUNDARY"))
        slide_vals = rhs_to_int_list(get_rhs_text("ALLOW_AGENT_SLIDING"))
        for i, cb in enumerate(self.cb_clamp):
            cb.blockSignals(True)
            cb.setChecked(bool(clamp_vals[i]))
            cb.blockSignals(False)
        for i, cb in enumerate(self.cb_slide):
            cb.blockSignals(True)
            cb.setChecked(bool(slide_vals[i]))
            cb.blockSignals(False)

        def rhs_is_true(rhs: Optional[str]) -> bool:
            if not rhs:
                return False
            return rhs.strip() == "True"

        self.cb_fibre.blockSignals(True)
        self.cb_diff.blockSignals(True)
        self.cb_cells.blockSignals(True)
        self.cb_fibre.setChecked(rhs_is_true(get_rhs_text("INCLUDE_FIBRE_NETWORK")))
        self.cb_diff.setChecked(rhs_is_true(get_rhs_text("INCLUDE_DIFFUSION")))
        self.cb_cells.setChecked(rhs_is_true(get_rhs_text("INCLUDE_CELLS")))
        self.cb_fibre.blockSignals(False)
        self.cb_diff.blockSignals(False)
        self.cb_cells.blockSignals(False)

        self.set_ui_enabled(True)
        self._editor_dirty = False
        self.log.append(f"Loaded: {path}")

    def apply_all_fields_to_editor(self):
        if not self.path or not self.lines:
            return

        try:
            # 1) scalars
            replace_var_line(self.lines, self.locs, "TIME_STEP", py_num_text(self.ed_time_step.text()))
            replace_var_line(self.lines, self.locs, "STEPS", py_num_text(self.ed_steps.text()))
            replace_var_line(self.lines, self.locs, "SAVE_EVERY_N_STEPS", py_num_text(self.ed_save_every.text()))

            # 2) boundary lists
            coords = [float(ed.text().strip() or 0) for ed in self.ed_boundary_coords]
            disp = [float(ed.text().strip() or 0) for ed in self.ed_boundary_disp]
            disp_par = [float(ed.text().strip() or 0) for ed in self.ed_boundary_disp_par]

            replace_var_line(self.lines, self.locs, "BOUNDARY_COORDS", format_list(coords, as_int=False))
            replace_var_line(self.lines, self.locs, "BOUNDARY_DISP_RATES", format_list(disp, as_int=False))
            replace_var_line(self.lines, self.locs, "BOUNDARY_DISP_RATES_PARALLEL", format_list(disp_par, as_int=False))
            clamp_vals = [1 if cb.isChecked() else 0 for cb in self.cb_clamp]
            slide_vals = [1 if cb.isChecked() else 0 for cb in self.cb_slide]
            replace_var_line(self.lines, self.locs, "CLAMP_AGENT_TOUCHING_BOUNDARY", format_list([float(x) for x in clamp_vals], as_int=True))
            replace_var_line(self.lines, self.locs, "ALLOW_AGENT_SLIDING", format_list([float(x) for x in slide_vals], as_int=True))

        except Exception as e:
            # Do not spam modal dialogs while typing, log instead
            self.log.append(f"Input error: {e}")
            return

        # Push updated file into editor without losing cursor position too much
        cur = self.editor.textCursor()
        pos = cur.position()
        vpos = self.editor.verticalScrollBar().value()
        hpos = self.editor.horizontalScrollBar().value()
        text = "".join(self.lines)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False

        if self.editor.hasFocus():
            cur2 = self.editor.textCursor()
            cur2.setPosition(min(pos, len(text)))
            self.editor.setTextCursor(cur2)
        else:
            self.editor.verticalScrollBar().setValue(vpos)
            self.editor.horizontalScrollBar().setValue(hpos)

    def on_toggle_bool(self, var: str, cb: QCheckBox, goto: bool):
        if not self.path:
            return
        try:
            replace_var_line(self.lines, self.locs, var, py_bool(cb.isChecked()))
        except Exception as e:
            self.log.append(f"Toggle error for {var}: {e}")
            return

        # Update editor text
        cur = self.editor.textCursor()
        pos = cur.position()
        vpos = self.editor.verticalScrollBar().value()
        hpos = self.editor.horizontalScrollBar().value()
        text = "".join(self.lines)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False
        if self.editor.hasFocus():
            cur2 = self.editor.textCursor()
            cur2.setPosition(min(pos, len(text)))
            self.editor.setTextCursor(cur2)
        else:
            self.editor.verticalScrollBar().setValue(vpos)
            self.editor.horizontalScrollBar().setValue(hpos)

        if goto:
            self.goto_var(var)

    def on_toggle_list(self, var: str, boxes: List[QCheckBox], goto: bool = False):
        if not self.path:
            return
        try:
            vals = [1 if cb.isChecked() else 0 for cb in boxes]
            replace_var_line(self.lines, self.locs, var, format_list([float(x) for x in vals], as_int=True))
        except Exception as e:
            self.log.append(f"Toggle error for {var}: {e}")
            return

        cur = self.editor.textCursor()
        pos = cur.position()
        vpos = self.editor.verticalScrollBar().value()
        hpos = self.editor.horizontalScrollBar().value()
        text = "".join(self.lines)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False
        if self.editor.hasFocus():
            cur2 = self.editor.textCursor()
            cur2.setPosition(min(pos, len(text)))
            self.editor.setTextCursor(cur2)
        else:
            self.editor.verticalScrollBar().setValue(vpos)
            self.editor.horizontalScrollBar().setValue(hpos)

        if goto:
            self.goto_var(var)

    def goto_var(self, var: str, keep_hscroll: bool = True):
        self._prepare_for_goto()
        doc = self.editor.document()
        pattern = QRegularExpression(rf"^\s*{re.escape(var)}\s*=.*$")
        pattern.setPatternOptions(QRegularExpression.MultilineOption)
        cursor = doc.find(pattern, 0)
        if cursor.isNull():
            return
        hbar = self.editor.horizontalScrollBar()
        hpos = hbar.value()
        vbar = self.editor.verticalScrollBar()
        self.editor.setTextCursor(cursor)
        vbar.setValue(cursor.blockNumber())
        self.editor.ensureCursorVisible()
        hbar.setValue(hpos if keep_hscroll else 0)

    def goto_subsection(self, title: str, keep_hscroll: bool = True):
        self._prepare_for_goto()
        doc = self.editor.document()
        pattern = QRegularExpression(rf"^\s*# \+\+\s*{re.escape(title)}\b.*$")
        pattern.setPatternOptions(QRegularExpression.MultilineOption)
        cursor = doc.find(pattern, 0)
        if cursor.isNull():
            return
        hbar = self.editor.horizontalScrollBar()
        hpos = hbar.value()
        vbar = self.editor.verticalScrollBar()
        self.editor.setTextCursor(cursor)
        vbar.setValue(cursor.blockNumber())
        self.editor.ensureCursorVisible()
        hbar.setValue(hpos if keep_hscroll else 0)

    def on_save(self):
        if not self.path:
            return
        # Apply any pending UI edits before saving
        if self._debounce.isActive():
            self._debounce.stop()
            self.apply_all_fields_to_editor()

        # Save editor contents (includes manual edits)
        editor_text = self.editor.toPlainText()
        self.lines = editor_text.splitlines(True)
        self.locs = build_var_index(self.lines, self.vars_needed)

        try:
            bak = self.path + ".bak"
            if os.path.exists(self.path):
                with open(self.path, "rb") as fsrc:
                    data = fsrc.read()
                with open(bak, "wb") as fdst:
                    fdst.write(data)

            with open(self.path, "w", encoding="utf-8") as f:
                f.write(editor_text)

            self.log.append(f"Saved. Backup: {bak}")
            self._editor_dirty = False
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _on_editor_text_changed(self):
        if self._suppress_editor_signal:
            return
        self._editor_dirty = True
        self._editor_debounce.start(200)

    def _refresh_from_editor(self):
        text = self.editor.toPlainText()
        self.lines = text.splitlines(True)
        self.locs = build_var_index(self.lines, self.vars_needed)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            widget_to_var = {
                self.ed_time_step: "TIME_STEP",
                self.ed_steps: "STEPS",
                self.ed_save_every: "SAVE_EVERY_N_STEPS",
            }
            if obj in widget_to_var:
                self.goto_var(widget_to_var[obj], keep_hscroll=True)
            elif obj in self.ed_boundary_coords:
                self.goto_var("BOUNDARY_COORDS", keep_hscroll=True)
            elif obj in self.ed_boundary_disp:
                self.goto_var("BOUNDARY_DISP_RATES", keep_hscroll=True)
            elif obj in self.ed_boundary_disp_par:
                self.goto_var("BOUNDARY_DISP_RATES_PARALLEL", keep_hscroll=True)
        return super().eventFilter(obj, event)

    def on_run(self):
        if not self.path:
            return

        # Save current state before run (optional, but usually desired)
        self.on_save()

        if self.process and self.process.state() != QProcess.NotRunning:
            self.log.append("Process already running.")
            return

        self.process = QProcess(self)
        workdir = os.path.dirname(self.path)
        venv_bat = "venv.bat"
        script_name = os.path.basename(self.path)
        if os.path.exists(os.path.join(workdir, venv_bat)):
            cmd = f'call {venv_bat} && python -u {script_name}'
            self.process.setProgram("cmd.exe")
            self.process.setArguments(["/c", cmd])
        else:
            self.process.setProgram(sys.executable)
            self.process.setArguments([self.path])
        self.process.setWorkingDirectory(workdir)

        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._proc_finished)

        if os.path.exists(os.path.join(workdir, venv_bat)):
            self.log.append(f"Running: {venv_bat} -> python -u {script_name}")
        else:
            self.log.append(f"Running: {sys.executable} {shlex.quote(self.path)}")
        self.process.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def on_stop(self):
        if not self.process:
            return
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            if not self.process.waitForFinished(800):
                self.process.kill()

    def _read_stdout(self):
        if not self.process:
            return
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.log.append(data.rstrip("\n"))

    def _read_stderr(self):
        if not self.process:
            return
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data:
            self.log.append(data.rstrip("\n"))

    def _proc_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        self.log.append(f"Finished. exit_code={exit_code}, status={exit_status}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Base, QColor(42, 42, 42))
    dark_palette.setColor(QPalette.AlternateBase, QColor(66, 66, 66))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
