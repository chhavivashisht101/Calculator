"""
Scientific Calculator - MCA Mini Project
=========================================
A full-featured scientific calculator with a graphical interface,
built using Python and Tkinter (no external libraries required).

Features:
  - Basic arithmetic: +, -, *, /, %, powers
  - Trigonometric functions: sin, cos, tan (DEG/RAD mode)
  - Inverse trig: asin, acos, atan
  - Hyperbolic: sinh, cosh, tanh
  - Logarithms: log (base 10), ln (natural), log2
  - Exponential, square root, cube root
  - Factorial (n!), absolute value, floor, ceil, round
  - Constants: pi, e
  - Parentheses for nested expressions
  - Calculation history (click to reuse)
  - Keyboard support
  - Memory functions (MC, MR, M+, M-)
  - DEG / RAD toggle

Author: MCA Student - AI & ML
Date: 2026
"""

import math
import tkinter as tk
from tkinter import font as tkFont
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────
#  Expression Evaluator (tokenizer + parser)
# ──────────────────────────────────────────────

class CalcError(Exception):
    """Raised when an expression cannot be evaluated."""
    pass

# Functions that take exactly one argument and don't depend on angle mode
SIMPLE_FUNCS = {
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log10,
    "ln": math.log,
    "log2": math.log2,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "abs": abs,
    "fact": None,  # handled separately for gamma support
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "sign": lambda x: math.copysign(1, x) if x != 0 else 0,
}

CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "phi": (1 + math.sqrt(5)) / 2,
}

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3}

def _factorial(n: float) -> float:
    """Factorial with gamma function support for non-integers."""
    if n < 0:
        raise CalcError("Cannot compute factorial of a negative number")
    if n == int(n):
        return float(math.factorial(int(n)))
    return math.gamma(n + 1)

def _tokenize(expr: str, deg_mode: bool) -> List[tuple]:
    """
    Convert expression string into a list of tokens.
    Each token is (type, value) where type is one of:
      'num', 'op', 'func', 'const', 'lparen', 'rparen', 'comma'
    """
    tokens = []
    i = 0
    s = expr.strip()

    while i < len(s):
        c = s[i]

        # Skip whitespace
        if c in (" ", "\t"):
            i += 1
            continue

        # Numbers (including decimals and scientific notation)
        if c.isdigit() or (c == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            num = ""
            while i < len(s) and (s[i].isdigit() or s[i] == "."):
                num += s[i]
                i += 1
            # Scientific notation: 1.5e-10
            if i < len(s) and s[i] in ("e", "E"):
                # Distinguish from constant 'e'
                if i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] in ("+", "-")):
                    num += s[i]
                    i += 1
                    if s[i] in ("+", "-"):
                        num += s[i]
                        i += 1
                    while i < len(s) and s[i].isdigit():
                        num += s[i]
                        i += 1
            try:
                tokens.append(("num", float(num)))
            except ValueError:
                raise CalcError(f"Invalid number: {num}")
            continue

        # Identifiers (functions or constants) — allow digits after first char (e.g. log2)
        if c.isalpha() or c == "_":
            name = ""
            while i < len(s) and (s[i].isalnum() or s[i] == "_"):
                name += s[i]
                i += 1
            name_lower = name.lower()
            if i < len(s) and s[i] == "(":
                tokens.append(("func", name_lower))
            elif name_lower in CONSTANTS:
                tokens.append(("const", name_lower))
            else:
                raise CalcError(f"Unknown identifier: {name}")
            continue

        # Operators
        if c in "+-*/%^":
            if c == "*" and i + 1 < len(s) and s[i + 1] == "*":
                tokens.append(("op", "^"))
                i += 2
            else:
                tokens.append(("op", c))
                i += 1
            continue

        # Parentheses and comma
        if c == "(":
            tokens.append(("lparen", c))
            i += 1
            continue
        if c == ")":
            tokens.append(("rparen", c))
            i += 1
            continue
        if c == ",":
            tokens.append(("comma", c))
            i += 1
            continue

        raise CalcError(f"Invalid character: '{c}'")

    return tokens

def _parse(tokens: List[tuple], deg_mode: bool) -> float:
    """
    Recursive-descent parser.
    Grammar:
      expr    → term (('+' | '-') term)*
      term    → factor (('*' | '/' | '%') factor)*
      factor  → power ('^' power)*      (right-associative)
      power   → unary
      unary   → ('-' | '+') unary | primary
      primary → NUMBER | CONST | func '(' expr (',' expr)* ')' | '(' expr ')'
    """
    pos = [0]  # mutable index so nested functions can advance it

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def consume():
        if pos[0] >= len(tokens):
            raise CalcError("Unexpected end of expression")
        tok = tokens[pos[0]]
        pos[0] += 1
        return tok

    def parse_expr():
        left = parse_term()
        while peek() and peek()[0] == "op" and peek()[1] in ("+", "-"):
            op = consume()[1]
            right = parse_term()
            if op == "+":
                left += right
            else:
                left -= right
        return left

    def parse_term():
        left = parse_factor()
        while peek() and peek()[0] == "op" and peek()[1] in ("*", "/", "%"):
            op = consume()[1]
            right = parse_factor()
            if op == "*":
                left *= right
            elif op == "/":
                if right == 0:
                    raise CalcError("Division by zero")
                left /= right
            else:
                left %= right
        return left

    def parse_factor():
        left = parse_unary()
        if peek() and peek()[0] == "op" and peek()[1] == "^":
            consume()
            right = parse_factor()  # right-associative
            left = math.pow(left, right)
        return left

    def parse_unary():
        tok = peek()
        if tok and tok[0] == "op" and tok[1] in ("-", "+"):
            op = consume()[1]
            val = parse_unary()
            return -val if op == "-" else val
        return parse_primary()

    def parse_primary():
        tok = consume()
        if tok is None:
            raise CalcError("Unexpected end of expression")

        ttype, tval = tok

        if ttype == "num":
            return tval

        if ttype == "const":
            return CONSTANTS[tval]

        if ttype == "lparen":
            val = parse_expr()
            close = peek()
            if close is None or close[0] != "rparen":
                raise CalcError("Missing closing parenthesis")
            consume()  # consume ')'
            return val

        if ttype == "func":
            open_paren = peek()
            if open_paren is None or open_paren[0] != "lparen":
                raise CalcError(f"Expected '(' after {tval}")
            consume()  # consume '('

            args = [parse_expr()]
            while peek() and peek()[0] == "comma":
                consume()
                args.append(parse_expr())

            close = peek()
            if close is None or close[0] != "rparen":
                raise CalcError("Missing closing parenthesis")
            consume()  # consume ')'

            return _call_function(tval, args, deg_mode)

        raise CalcError(f"Unexpected token: {tval}")

    result = parse_expr()
    if pos[0] < len(tokens):
        raise CalcError("Unexpected trailing input")
    return result

def _call_function(name: str, args: List[float], deg_mode: bool) -> float:
    """Dispatch a function call with the given arguments."""
    deg = deg_mode

    # Trigonometric (angle-mode aware)
    if name == "sin":
        x = math.radians(args[0]) if deg else args[0]
        return math.sin(x)
    if name == "cos":
        x = math.radians(args[0]) if deg else args[0]
        return math.cos(x)
    if name == "tan":
        x = math.radians(args[0]) if deg else args[0]
        return math.tan(x)
    if name == "asin":
        r = math.asin(args[0])
        return math.degrees(r) if deg else r
    if name == "acos":
        r = math.acos(args[0])
        return math.degrees(r) if deg else r
    if name == "atan":
        r = math.atan(args[0])
        return math.degrees(r) if deg else r

    # Factorial / gamma
    if name == "fact":
        return _factorial(args[0])

    # Simple single-arg functions
    if name in SIMPLE_FUNCS and SIMPLE_FUNCS[name] is not None:
        return SIMPLE_FUNCS[name](args[0])

    # Multi-arg functions
    if name == "pow":
        return math.pow(args[0], args[1])
    if name == "mod":
        return args[0] % args[1]
    if name == "max":
        return max(args)
    if name == "min":
        return min(args)
    if name == "gcd":
        result = int(abs(args[0]))
        for a in args[1:]:
            result = math.gcd(result, int(abs(a)))
        return float(result)
    if name == "inv":
        if args[0] == 0:
            raise CalcError("Division by zero")
        return 1 / args[0]
    if name == "random":
        import random
        return random.random()

    raise CalcError(f"Unknown function: {name}")

def evaluate(expression: str, deg_mode: bool = True) -> float:
    """
    Evaluate a mathematical expression string and return the result.
    Raises CalcError on invalid input.
    """
    if not expression or not expression.strip():
        raise CalcError("Empty expression")
    tokens = _tokenize(expression, deg_mode)
    if not tokens:
        raise CalcError("Empty expression")
    return _parse(tokens, deg_mode)

def format_result(value: float) -> str:
    """Format a numeric result for display."""
    if math.isnan(value):
        return "Error"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val < 1e-6 or abs_val >= 1e15:
        return f"{value:.8e}"
    # Round to 12 significant digits, strip trailing zeros
    rounded = round(value, 10)
    if rounded == int(rounded):
        return str(int(rounded))
    s = f"{rounded:.10f}".rstrip("0").rstrip(".")
    return s

# ──────────────────────────────────────────────
#  Calculator Application (Tkinter GUI)
# ──────────────────────────────────────────────

# Color palette — dark theme
COLORS = {
    "bg":           "#0f172a",   # slate-950
    "panel":        "#1e293b",   # slate-800
    "display_bg":   "#0f172a",
    "expr_color":   "#94a3b8",   # slate-400
    "result_color": "#f8fafc",   # slate-50
    "num_bg":       "#334155",   # slate-700
    "num_fg":       "#f8fafc",
    "num_hover":    "#475569",
    "op_bg":        "#78350f",   # amber-900-ish
    "op_fg":        "#fcd34d",   # amber-300
    "op_hover":     "#92400e",
    "func_bg":      "#0c4a6e",   # sky-900-ish
    "func_fg":      "#7dd3fc",   # sky-300
    "func_hover":   "#075985",
    "const_bg":     "#1e1b4b",   # indigo-ish (used sparingly, not violet-heavy)
    "const_fg":     "#a5b4fc",
    "const_hover":  "#312e81",
    "eq_bg":        "#0284c7",   # sky-600
    "eq_fg":        "#ffffff",
    "eq_hover":     "#0ea5e9",
    "clear_bg":     "#7f1d1d",
    "clear_fg":     "#fca5a5",
    "clear_hover":  "#991b1b",
    "del_bg":       "#44403c",
    "del_fg":       "#e7e5e4",
    "del_hover":    "#57534e",
    "history_bg":   "#1e293b",
    "error_color":  "#f87171",
}

@dataclass
class HistoryEntry:
    expression: str
    result: str

class ScientificCalculator:
    """Main calculator application class."""

    # Button layout: (label, insert_value, button_type)
    # 'action' buttons have special handling
    BUTTONS = [
        # Row 0 — scientific functions
        ("sin",   "sin(",   "func"),
        ("cos",   "cos(",   "func"),
        ("tan",   "tan(",   "func"),
        ("π",     "pi",     "const"),
        ("e",     "e",      "const"),
        ("xʸ",    "^",      "op"),
        ("x!",    "fact(",  "func"),

        # Row 1
        ("asin",  "asin(",  "func"),
        ("acos",  "acos(",  "func"),
        ("atan",  "atan(",  "func"),
        ("log",   "log(",   "func"),
        ("ln",    "ln(",    "func"),
        ("√",     "sqrt(",  "func"),
        ("∛",     "cbrt(",  "func"),

        # Row 2
        ("sinh",  "sinh(",  "func"),
        ("cosh",  "cosh(",  "func"),
        ("tanh",  "tanh(",  "func"),
        ("exp",   "exp(",   "func"),
        ("|x|",   "abs(",   "func"),
        ("mod",   " mod ",  "op"),
        ("1/x",   "1/",     "func"),

        # Row 3
        ("(",     "(",      "paren"),
        (")",     ")",      "paren"),
        ("x²",    "^2",     "op"),
        ("10ˣ",   "10^",    "func"),
        ("RND",   "random()", "func"),
        ("DEG",   "DEG",    "action"),
        ("RAD",   "RAD",    "action"),

        # Row 4 — numeric + control
        ("7",     "7",      "num"),
        ("8",     "8",      "num"),
        ("9",     "9",      "num"),
        ("DEL",   "DEL",    "action"),
        ("CLR",   "CLR",    "action"),
        ("%",     "%",      "op"),
        ("÷",     "/",      "op"),

        # Row 5
        ("4",     "4",      "num"),
        ("5",     "5",      "num"),
        ("6",     "6",      "num"),
        ("×",     "*",      "op"),
        ("−",     "-",      "op"),

        # Row 6
        ("1",     "1",      "num"),
        ("2",     "2",      "num"),
        ("3",     "3",      "num"),
        ("+",     "+",      "op"),

        # Row 7
        ("0",     "0",      "num"),
        (".",     ".",      "num"),
        ("=",     "=",      "action"),
    ]

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Scientific Calculator — MCA Mini Project")
        self.root.geometry("440x720")
        self.root.minsize(400, 680)
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        # State
        self.expression = ""
        self.deg_mode = True
        self.history: List[HistoryEntry] = []
        self.memory: float = 0.0
        self.last_result: str = "0"
        self.error_state = False

        self._build_ui()
        self._bind_keyboard()

    # ─── UI Construction ───

    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self.root, bg=COLORS["panel"], height=52)
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        header.pack_propagate(False)

        title_font = tkFont.Font(size=13, weight="bold")
        title = tk.Label(header, text="SciCalc", font=title_font,
                         bg=COLORS["panel"], fg=COLORS["result_color"])
        title.pack(side=tk.LEFT, padx=12, pady=8)

        sub_font = tkFont.Font(size=8)
        sub = tk.Label(header, text="Scientific Calculator", font=sub_font,
                       bg=COLORS["panel"], fg=COLORS["expr_color"])
        sub.pack(side=tk.LEFT, pady=(8, 0))

        # Right-side buttons: DEG/RAD indicator, History
        self.angle_btn = tk.Button(
            header, text="DEG", width=5, font=tkFont.Font(size=9, weight="bold"),
            bg=COLORS["func_bg"], fg=COLORS["func_fg"],
            activebackground=COLORS["func_hover"], activeforeground=COLORS["func_fg"],
            relief="flat", bd=0, padx=8, pady=2,
            command=self._toggle_angle_mode,
        )
        self.angle_btn.pack(side=tk.RIGHT, padx=(0, 6), pady=8)

        hist_btn = tk.Button(
            header, text="History", width=7, font=tkFont.Font(size=8),
            bg=COLORS["num_bg"], fg=COLORS["result_color"],
            activebackground=COLORS["num_hover"], activeforeground=COLORS["result_color"],
            relief="flat", bd=0, padx=6, pady=2,
            command=self._show_history,
        )
        hist_btn.pack(side=tk.RIGHT, padx=(0, 4), pady=8)

        # ---- Display ----
        display_frame = tk.Frame(self.root, bg=COLORS["display_bg"])
        display_frame.pack(fill=tk.X, padx=8, pady=(4, 8))

        self.expr_label = tk.Label(
            display_frame, text="", anchor="e", justify="right",
            font=tkFont.Font(size=12), bg=COLORS["display_bg"],
            fg=COLORS["expr_color"], wraplength=400,
        )
        self.expr_label.pack(fill=tk.X, padx=12, pady=(8, 0))

        self.result_label = tk.Label(
            display_frame, text="0", anchor="e", justify="right",
            font=tkFont.Font(size=28, weight="normal"),
            bg=COLORS["display_bg"], fg=COLORS["result_color"],
            wraplength=400,
        )
        self.result_label.pack(fill=tk.X, padx=12, pady=(2, 12))

        # ---- Memory row ----
        mem_frame = tk.Frame(self.root, bg=COLORS["bg"])
        mem_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        for label, cmd in [("MC", self._mem_clear), ("MR", self._mem_recall),
                           ("M+", self._mem_add), ("M-", self._mem_subtract)]:
            btn = tk.Button(
                mem_frame, text=label, width=4,
                font=tkFont.Font(size=8, weight="bold"),
                bg=COLORS["del_bg"], fg=COLORS["del_fg"],
                activebackground=COLORS["del_hover"], activeforeground=COLORS["del_fg"],
                relief="flat", bd=0, padx=4, pady=3,
                command=cmd,
            )
            btn.pack(side=tk.LEFT, padx=2)

        self.mem_indicator = tk.Label(
            mem_frame, text="", font=tkFont.Font(size=8),
            bg=COLORS["bg"], fg=COLORS["func_fg"],
        )
        self.mem_indicator.pack(side=tk.RIGHT, padx=8)

        # ---- Keypad ----
        keypad = tk.Frame(self.root, bg=COLORS["bg"])
        keypad.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # 7 columns
        for idx, (label, value, btype) in enumerate(self.BUTTONS):
            row = idx // 7
            col = idx % 7

            # Span '=' across 2 columns (last row)
            colspan = 1
            if label == "=":
                colspan = 2

            btn = self._make_button(keypad, label, value, btype)
            btn.grid(
                row=row, column=col, columnspan=colspan,
                sticky="nsew", padx=3, pady=3,
            )

        # Make rows and columns expand proportionally
        for r in range(8):
            keypad.grid_rowconfigure(r, weight=1)
        for c in range(7):
            keypad.grid_columnconfigure(c, weight=1)

    def _make_button(self, parent, label, value, btype):
        """Create a styled calculator button."""
        bg, fg, hover = self._button_colors(btype, label)

        btn = tk.Button(
            parent, text=label if label not in ("DEG", "RAD") else (
                "DEG" if self.deg_mode else "RAD"),
            font=tkFont.Font(size=10, weight="bold"),
            bg=bg, fg=fg,
            activebackground=hover, activeforeground=fg,
            relief="flat", bd=0, padx=4, pady=8,
            command=lambda v=value, l=label: self._on_button(v, l),
        )

        # Hover effect
        btn.bind("<Enter>", lambda e, b=btn, h=hover: b.configure(bg=h))
        btn.bind("<Leave>", lambda e, b=btn, base=bg: b.configure(bg=base))

        return btn

    def _button_colors(self, btype, label):
        if label == "=":
            return COLORS["eq_bg"], COLORS["eq_fg"], COLORS["eq_hover"]
        if label == "CLR":
            return COLORS["clear_bg"], COLORS["clear_fg"], COLORS["clear_hover"]
        if label == "DEL":
            return COLORS["del_bg"], COLORS["del_fg"], COLORS["del_hover"]
        if label in ("DEG", "RAD"):
            return COLORS["func_bg"], COLORS["func_fg"], COLORS["func_hover"]
        mapping = {
            "num":   (COLORS["num_bg"],   COLORS["num_fg"],   COLORS["num_hover"]),
            "op":    (COLORS["op_bg"],    COLORS["op_fg"],    COLORS["op_hover"]),
            "func":  (COLORS["func_bg"],  COLORS["func_fg"],  COLORS["func_hover"]),
            "const": (COLORS["const_bg"], COLORS["const_fg"], COLORS["const_hover"]),
            "paren": (COLORS["num_bg"],   COLORS["num_fg"],   COLORS["num_hover"]),
        }
        return mapping.get(btype, mapping["num"])

    # ─── Event Handlers ───

    def _on_button(self, value, label):
        """Handle button presses."""
        self.error_state = False

        if value == "DEG" or value == "RAD":
            self._toggle_angle_mode()
            return

        if value == "DEL":
            self.expression = self.expression[:-1]
            self._update_display()
            return

        if value == "CLR":
            self.expression = ""
            self.last_result = "0"
            self._update_display()
            return

        if value == "=":
            self._evaluate()
            return

        self.expression += value
        self._update_display()
        self._live_preview()

    def _evaluate(self):
        """Evaluate the current expression and store in history."""
        if not self.expression.strip():
            return
        try:
            result = evaluate(self.expression, self.deg_mode)
            display = format_result(result)
            if display == "Error":
                self.error_state = True
                self.result_label.config(text="Error", fg=COLORS["error_color"])
                return

            # Save to history
            self.history.append(HistoryEntry(self.expression, display))
            self.last_result = display
            self.expression = display
            self._update_display()
            self.result_label.config(fg=COLORS["result_color"])

        except CalcError as e:
            self.error_state = True
            self.result_label.config(text=f"Error: {e}", fg=COLORS["error_color"])
        except Exception:
            self.error_state = True
            self.result_label.config(text="Error", fg=COLORS["error_color"])

    def _live_preview(self):
        """Show a live result preview as the user types."""
        if not self.expression.strip():
            self.result_label.config(text="0", fg=COLORS["result_color"])
            return
        try:
            result = evaluate(self.expression, self.deg_mode)
            display = format_result(result)
            if display != "Error":
                self.result_label.config(text=display, fg=COLORS["expr_color"])
        except Exception:
            pass  # Don't show error during typing

    def _update_display(self):
        """Refresh the expression and result labels."""
        self.expr_label.config(text=self.expression or "")
        if not self.error_state:
            self.result_label.config(text=self.last_result, fg=COLORS["result_color"])

    def _toggle_angle_mode(self):
        self.deg_mode = not self.deg_mode
        self.angle_btn.config(text="DEG" if self.deg_mode else "RAD")
        self._live_preview()

    # ─── Memory ───

    def _mem_clear(self):
        self.memory = 0.0
        self.mem_indicator.config(text="")

    def _mem_recall(self):
        self.expression += str(self.memory) if self.memory == int(self.memory) else f"{self.memory}"
        self._update_display()
        self._live_preview()

    def _mem_add(self):
        try:
            val = evaluate(self.expression, self.deg_mode) if self.expression.strip() else 0
            self.memory += val
            self.mem_indicator.config(text=f"M = {format_result(self.memory)}")
        except Exception:
            pass

    def _mem_subtract(self):
        try:
            val = evaluate(self.expression, self.deg_mode) if self.expression.strip() else 0
            self.memory -= val
            self.mem_indicator.config(text=f"M = {format_result(self.memory)}")
        except Exception:
            pass

    # ─── History Window ───

    def _show_history(self):
        """Open a history window."""
        win = tk.Toplevel(self.root)
        win.title("Calculation History")
        win.geometry("380x500")
        win.configure(bg=COLORS["history_bg"])
        win.transient(self.root)
        win.grab_set()

        header = tk.Frame(win, bg=COLORS["panel"], height=44)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="History", font=tkFont.Font(size=11, weight="bold"),
            bg=COLORS["panel"], fg=COLORS["result_color"],
        ).pack(side=tk.LEFT, padx=12, pady=8)

        count_label = tk.Label(
            header, text=f"({len(self.history)})", font=tkFont.Font(size=9),
            bg=COLORS["panel"], fg=COLORS["expr_color"],
        )
        count_label.pack(side=tk.LEFT, pady=(8, 0))

        if self.history:
            clear_btn = tk.Button(
                header, text="Clear All", font=tkFont.Font(size=8),
                bg=COLORS["clear_bg"], fg=COLORS["clear_fg"],
                activebackground=COLORS["clear_hover"], activeforeground=COLORS["clear_fg"],
                relief="flat", bd=0, padx=8, pady=2,
                command=lambda: self._clear_history(win),
            )
            clear_btn.pack(side=tk.RIGHT, padx=8, pady=8)

        # Scrollable area
        container = tk.Frame(win, bg=COLORS["history_bg"])
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        canvas = tk.Canvas(container, bg=COLORS["history_bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["history_bg"])

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        if not self.history:
            tk.Label(
                scroll_frame, text="No calculations yet\n\nYour history will appear here",
                font=tkFont.Font(size=10), bg=COLORS["history_bg"],
                fg=COLORS["expr_color"], justify="center",
            ).pack(pady=60)
        else:
            for entry in reversed(self.history):
                card = tk.Frame(
                    scroll_frame, bg=COLORS["num_bg"],
                    highlightbackground=COLORS["panel"], highlightthickness=1,
                )
                card.pack(fill=tk.X, pady=3, padx=4)

                expr_lbl = tk.Label(
                    card, text=entry.expression, anchor="w", justify="left",
                    font=tkFont.Font(size=9), bg=COLORS["num_bg"],
                    fg=COLORS["expr_color"], wraplength=340,
                )
                expr_lbl.pack(fill=tk.X, padx=8, pady=(6, 0))

                res_lbl = tk.Label(
                    card, text=f"= {entry.result}", anchor="w",
                    font=tkFont.Font(size=14), bg=COLORS["num_bg"],
                    fg=COLORS["result_color"],
                )
                res_lbl.pack(fill=tk.X, padx=8, pady=(0, 6))

                # Click to reuse
                def make_reuse(e=entry, w=win):
                    return lambda: self._reuse_history(e, w)
                for widget in (card, expr_lbl, res_lbl):
                    widget.bind("<Button-1>", make_reuse())
                    widget.configure(cursor="hand2")

    def _reuse_history(self, entry, win):
        self.expression = entry.result
        self.last_result = entry.result
        self._update_display()
        win.destroy()

    def _clear_history(self, win):
        self.history.clear()
        win.destroy()
        self._show_history()

    # ─── Keyboard Bindings ───

    def _bind_keyboard(self):
        self.root.bind("<Return>", lambda e: self._evaluate())
        self.root.bind("<KP_Enter>", lambda e: self._evaluate())
        self.root.bind("<BackSpace>", lambda e: self._on_button("DEL", "DEL"))
        self.root.bind("<Escape>", lambda e: self._on_button("CLR", "CLR"))

        for key in "0123456789.+-*/()%^":
            self.root.bind(key, lambda e, k=key: self._on_key(k))
            if key.upper() != key:
                self.root.bind(key.upper(), lambda e, k=key: self._on_key(k))

    def _on_key(self, key):
        self.error_state = False
        self.expression += key
        self._update_display()
        self._live_preview()

    # ─── Main Loop ───

    def run(self):
        self.root.mainloop()

# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = ScientificCalculator()
    app.run()
