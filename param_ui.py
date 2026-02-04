import os
import re
import sys
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

from PySide6.QtCore import Qt, QTimer, QProcess, QEvent
from PySide6.QtCore import Qt, QTimer, QProcess, QEvent, QRegularExpression
from PySide6.QtGui import QColor, QFont, QTextCursor, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QPlainTextEdit, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QPushButton, QLineEdit,
    QCheckBox, QLabel, QMessageBox, QSplitter
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

        self.f_comment = fmt("#8a8a8a", italic=True)
        self.f_string = fmt("#ce9178")
        self.f_number = fmt("#b5cea8")
        self.f_keyword = fmt("#569cd6", bold=True)
        self.f_builtin = fmt("#4ec9b0")
        self.f_caps = fmt("#dcdcaa", bold=True)

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

    def highlightBlock(self, text: str):
        # Comments
        m = self.re_comment.search(text)
        if m:
            start, end = m.start(), m.end()
            self.setFormat(start, end - start, self.f_comment)
            code_part = text[:start]
        else:
            code_part = text

        # Strings
        for m in self.re_string.finditer(code_part):
            self.setFormat(m.start(), m.end() - m.start(), self.f_string)

        # Numbers
        for m in self.re_number.finditer(code_part):
            self.setFormat(m.start(), m.end() - m.start(), self.f_number)

        # Keywords and builtins
        for m in self.re_keyword.finditer(code_part):
            self.setFormat(m.start(), m.end() - m.start(), self.f_keyword)
        for m in self.re_builtin.finditer(code_part):
            self.setFormat(m.start(), m.end() - m.start(), self.f_builtin)

        # ALL_CAPS variables
        for m in self.re_caps.finditer(code_part):
            self.setFormat(m.start(), m.end() - m.start(), self.f_caps)


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
        self.setWindowTitle("Simulation Parameter Editor")

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

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))

        # Controls
        top = QHBoxLayout()
        btn_open = QPushButton("Open")
        btn_save = QPushButton("Save")
        btn_reload = QPushButton("Reload")
        btn_run = QPushButton("Run")
        btn_stop = QPushButton("Stop")
        btn_stop.setEnabled(False)

        top.addWidget(btn_open)
        top.addWidget(btn_save)
        top.addWidget(btn_reload)
        top.addStretch(1)
        top.addWidget(btn_run)
        top.addWidget(btn_stop)

        btn_open.clicked.connect(self.on_open)
        btn_save.clicked.connect(self.on_save)
        btn_reload.clicked.connect(self.on_reload)
        btn_run.clicked.connect(self.on_run)
        btn_stop.clicked.connect(self.on_stop)

        self.btn_save = btn_save
        self.btn_run = btn_run
        self.btn_stop = btn_stop

        # Parameter panels
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # 1) GLOBAL PARAMETERS
        gb_global = QGroupBox("1) GLOBAL PARAMETERS")
        fl_global = QFormLayout(gb_global)
        self.ed_time_step = QLineEdit()
        self.ed_steps = QLineEdit()
        self.ed_save_every = QLineEdit()
        fl_global.addRow("TIME_STEP", self.ed_time_step)
        fl_global.addRow("STEPS", self.ed_steps)
        fl_global.addRow("SAVE_EVERY_N_STEPS", self.ed_save_every)
        self.btn_goto_global = QPushButton("Go to section")
        fl_global.addRow("", self.btn_goto_global)

        # 2) BOUNDARY BEHAVIOUR
        gb_boundary = QGroupBox("2) BOUNDARY BEHAVIOUR")
        fl_boundary = QFormLayout(gb_boundary)
        self.ed_boundary_coords = QLineEdit()
        self.ed_boundary_disp = QLineEdit()
        self.ed_boundary_disp_par = QLineEdit()
        self.ed_clamp = QLineEdit()
        self.ed_slide = QLineEdit()
        fl_boundary.addRow("BOUNDARY_COORDS (csv)", self.ed_boundary_coords)
        fl_boundary.addRow("BOUNDARY_DISP_RATES (csv)", self.ed_boundary_disp)
        fl_boundary.addRow("BOUNDARY_DISP_RATES_PARALLEL (csv)", self.ed_boundary_disp_par)
        fl_boundary.addRow("CLAMP_AGENT_TOUCHING_BOUNDARY (0/1 csv)", self.ed_clamp)
        fl_boundary.addRow("ALLOW_AGENT_SLIDING (0/1 csv)", self.ed_slide)
        self.btn_goto_boundary = QPushButton("Go to section")
        fl_boundary.addRow("", self.btn_goto_boundary)

        # 3) FIBRE NETWORK
        gb_fibre = QGroupBox("3) FIBRE NETWORK")
        vb_fibre = QVBoxLayout(gb_fibre)
        self.cb_fibre = QCheckBox("INCLUDE_FIBRE_NETWORK")
        self.btn_goto_fibre = QPushButton("Go to section")
        row_fibre = QHBoxLayout()
        row_fibre.addWidget(self.cb_fibre)
        row_fibre.addStretch(1)
        row_fibre.addWidget(self.btn_goto_fibre)
        vb_fibre.addLayout(row_fibre)

        # 4) SPECIES DIFFUSION
        gb_diff = QGroupBox("4) SPECIES DIFFUSION")
        vb_diff = QVBoxLayout(gb_diff)
        self.cb_diff = QCheckBox("INCLUDE_DIFFUSION")
        self.btn_goto_diff = QPushButton("Go to section")
        row_diff = QHBoxLayout()
        row_diff.addWidget(self.cb_diff)
        row_diff.addStretch(1)
        row_diff.addWidget(self.btn_goto_diff)
        vb_diff.addLayout(row_diff)

        # 5) CELLS
        gb_cells = QGroupBox("5) CELLS")
        vb_cells = QVBoxLayout(gb_cells)
        self.cb_cells = QCheckBox("INCLUDE_CELLS")
        self.btn_goto_cells = QPushButton("Go to section")
        row_cells = QHBoxLayout()
        row_cells.addWidget(self.cb_cells)
        row_cells.addStretch(1)
        row_cells.addWidget(self.btn_goto_cells)
        vb_cells.addLayout(row_cells)

        panel_layout.addWidget(gb_global)
        panel_layout.addWidget(gb_boundary)
        panel_layout.addWidget(gb_fibre)
        panel_layout.addWidget(gb_diff)
        panel_layout.addWidget(gb_cells)
        panel_layout.addStretch(1)

        # Split view: left parameters, right editor, bottom log
        splitter_h = QSplitter(Qt.Horizontal)
        splitter_h.addWidget(panel)
        splitter_h.addWidget(self.editor)
        splitter_h.setStretchFactor(0, 0)
        splitter_h.setStretchFactor(1, 1)

        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(splitter_h)
        splitter_v.addWidget(self.log)
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 1)

        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(splitter_v)

        # Dark theme (simple inline stylesheet)
        self.apply_dark_theme()

        # Wire auto-update
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.apply_all_fields_to_editor)

        self._editor_debounce = QTimer()
        self._editor_debounce.setSingleShot(True)
        self._editor_debounce.timeout.connect(self._refresh_from_editor)

        for w in [
            self.ed_time_step, self.ed_steps, self.ed_save_every,
            self.ed_boundary_coords, self.ed_boundary_disp, self.ed_boundary_disp_par,
            self.ed_clamp, self.ed_slide
        ]:
            w.textChanged.connect(self._debounced_apply)
            w.installEventFilter(self)

        self.cb_fibre.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_FIBRE_NETWORK", self.cb_fibre, goto=True))
        self.cb_diff.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_DIFFUSION", self.cb_diff, goto=True))
        self.cb_cells.stateChanged.connect(lambda _: self.on_toggle_bool("INCLUDE_CELLS", self.cb_cells, goto=True))

        self.btn_goto_global.clicked.connect(lambda: self.goto_var("TIME_STEP"))
        self.btn_goto_boundary.clicked.connect(lambda: self.goto_var("BOUNDARY_COORDS"))
        self.btn_goto_fibre.clicked.connect(lambda: self.goto_var("INCLUDE_FIBRE_NETWORK"))
        self.btn_goto_diff.clicked.connect(lambda: self.goto_var("INCLUDE_DIFFUSION"))
        self.btn_goto_cells.clicked.connect(lambda: self.goto_var("INCLUDE_CELLS"))

        self.editor.textChanged.connect(self._on_editor_text_changed)

        self.set_ui_enabled(False)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget { background: #1e1e1e; color: #d4d4d4; }
            QGroupBox { border: 1px solid #3a3a3a; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
            QLineEdit, QPlainTextEdit, QTextEdit {
                background: #252526; border: 1px solid #3a3a3a; selection-background-color: #264f78;
            }
            QPushButton { background: #2d2d30; border: 1px solid #3a3a3a; padding: 6px 10px; }
            QPushButton:hover { background: #3a3a3a; }
            QPushButton:disabled { color: #777777; }
            QCheckBox::indicator { width: 14px; height: 14px; }
        """)

    def set_ui_enabled(self, enabled: bool):
        for w in [
            self.ed_time_step, self.ed_steps, self.ed_save_every,
            self.ed_boundary_coords, self.ed_boundary_disp, self.ed_boundary_disp_par,
            self.ed_clamp, self.ed_slide,
            self.cb_fibre, self.cb_diff, self.cb_cells,
            self.btn_goto_global, self.btn_goto_boundary,
            self.btn_goto_fibre, self.btn_goto_diff, self.btn_goto_cells,
            self.btn_save, self.btn_run
        ]:
            w.setEnabled(enabled)

    def _debounced_apply(self):
        # debounce to avoid rewriting the editor on every keystroke
        self._debounce.start(150)

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

        self.ed_boundary_coords.setText(list_to_csv(get_rhs_text("BOUNDARY_COORDS")))
        self.ed_boundary_disp.setText(list_to_csv(get_rhs_text("BOUNDARY_DISP_RATES")))
        self.ed_boundary_disp_par.setText(list_to_csv(get_rhs_text("BOUNDARY_DISP_RATES_PARALLEL")))
        self.ed_clamp.setText(list_to_csv(get_rhs_text("CLAMP_AGENT_TOUCHING_BOUNDARY")))
        self.ed_slide.setText(list_to_csv(get_rhs_text("ALLOW_AGENT_SLIDING")))

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
            coords = parse_csv_numbers(self.ed_boundary_coords.text())
            disp = parse_csv_numbers(self.ed_boundary_disp.text())
            disp_par = parse_csv_numbers(self.ed_boundary_disp_par.text())

            clamp = parse_csv_int01(self.ed_clamp.text())
            slide = parse_csv_int01(self.ed_slide.text())

            replace_var_line(self.lines, self.locs, "BOUNDARY_COORDS", format_list(coords, as_int=False))
            replace_var_line(self.lines, self.locs, "BOUNDARY_DISP_RATES", format_list(disp, as_int=False))
            replace_var_line(self.lines, self.locs, "BOUNDARY_DISP_RATES_PARALLEL", format_list(disp_par, as_int=False))
            replace_var_line(self.lines, self.locs, "CLAMP_AGENT_TOUCHING_BOUNDARY", format_list([float(x) for x in clamp], as_int=True))
            replace_var_line(self.lines, self.locs, "ALLOW_AGENT_SLIDING", format_list([float(x) for x in slide], as_int=True))

        except Exception as e:
            # Do not spam modal dialogs while typing, log instead
            self.log.append(f"Input error: {e}")
            return

        # Push updated file into editor without losing cursor position too much
        cur = self.editor.textCursor()
        pos = cur.position()
        text = "".join(self.lines)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False

        cur2 = self.editor.textCursor()
        cur2.setPosition(min(pos, len(text)))
        self.editor.setTextCursor(cur2)

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
        text = "".join(self.lines)
        self._suppress_editor_signal = True
        self.editor.setPlainText(text)
        self._suppress_editor_signal = False
        cur2 = self.editor.textCursor()
        cur2.setPosition(min(pos, len(text)))
        self.editor.setTextCursor(cur2)

        if goto:
            self.goto_var(var)

    def goto_var(self, var: str):
        doc = self.editor.document()
        pattern = QRegularExpression(rf"^\s*{re.escape(var)}\s*=.*$")
        cursor = doc.find(pattern, 0)
        if cursor.isNull():
            return
        self.editor.setTextCursor(cursor)
        rect = self.editor.cursorRect(cursor)
        sb = self.editor.verticalScrollBar()
        sb.setValue(sb.value() + rect.top())

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
                self.ed_boundary_coords: "BOUNDARY_COORDS",
                self.ed_boundary_disp: "BOUNDARY_DISP_RATES",
                self.ed_boundary_disp_par: "BOUNDARY_DISP_RATES_PARALLEL",
                self.ed_clamp: "CLAMP_AGENT_TOUCHING_BOUNDARY",
                self.ed_slide: "ALLOW_AGENT_SLIDING",
            }
            if obj in widget_to_var:
                self.goto_var(widget_to_var[obj])
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
        self.process.setProgram(sys.executable)
        self.process.setArguments([self.path])
        self.process.setWorkingDirectory(os.path.dirname(self.path))

        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._proc_finished)

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
    w = MainWindow()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
