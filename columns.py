import os
import re
import math
import importlib.util
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

###############################################
# Import attribute_mapping and roles_data automatically
###############################################

try:
    from attribute_mapping import attribute_mapping
except ImportError:
    attribute_mapping = {}

try:
    from roles_data import roles_data
except ImportError:
    roles_data = {}

###############################################
# Parsing utilities for the stats HTML
###############################################

_range_pat = re.compile(r"^\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*$")
_height_pat = re.compile(r"^(\d+)'\s*(\d+)(?:\"|”)?$")
_money_pat = re.compile(r"£\s*([\d.,]+)\s*([KkMm])?")


def parse_attr_range(value: str) -> Tuple[int, int, float]:
    if value is None:
        return 0, 0, 0.0
    s = str(value).strip()
    if not s or s == "-":
        return 0, 0, 0.0
    m = _range_pat.match(s)
    if not m:
        try:
            x = int(re.sub(r"[^0-9]", "", s))
            return x, x, float(x)
        except Exception:
            return 0, 0, 0.0
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return lo, hi, (lo + hi) / 2.0


def parse_height_to_cm(s: str) -> float:
    if not s:
        return float("nan")
    s = s.strip()
    m = _height_pat.match(s)
    if not m:
        return float("nan")
    feet, inches = int(m.group(1)), int(m.group(2))
    total_inches = feet * 12 + inches
    return round(total_inches * 2.54, 1)


def _money_to_number(part: str) -> float:
    part = part.strip()
    m = _money_pat.search(part)
    if not m:
        return float("nan")
    num = float(m.group(1).replace(",", ""))
    suf = (m.group(2) or "").lower()
    if suf == "k":
        num *= 1_000
    elif suf == "m":
        num *= 1_000_000
    return num


def parse_value_range_gbp(s: str) -> Tuple[float, float]:
    if not s or s.strip().lower() in {"unknown", "n/a", "-"}:
        return float("nan"), float("nan")
    if "-" in s:
        left, right = s.split("-", 1)
        return _money_to_number(left), _money_to_number(right)
    else:
        v = _money_to_number(s)
        return v, v

###############################################
# Data loading & normalization
###############################################

def load_stats_html(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path, flavor="bs4")
    if not tables:
        raise ValueError("No tables found in stats HTML")
    df = tables[0]

    cols = list(df.columns)
    nat_indices = [i for i, c in enumerate(cols) if str(c).strip().lower() == "nat"]
    if nat_indices:
        first_nat = nat_indices[0]
        cols[first_nat] = "Nationality"
        df.columns = cols

    if "Height" in df.columns:
        df["Height_cm"] = df["Height"].astype(str).map(parse_height_to_cm)
    if "Weight" in df.columns:
        df["Weight_kg"] = pd.to_numeric(df["Weight"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    if "Transfer Value" in df.columns:
        v = df["Transfer Value"].astype(str).map(parse_value_range_gbp)
        df["ValueMin_GBP"], df["ValueMax_GBP"] = zip(*v)

    bio_like = {"Rec", "Inf", "Name", "Position", "Nationality", "Height", "Weight", "Age", "Transfer Value",
                "Height_cm", "Weight_kg", "ValueMin_GBP", "ValueMax_GBP"}
    attr_cols: List[str] = [c for c in df.columns if c not in bio_like]

    for c in attr_cols:
        mins, maxs, means = zip(*(parse_attr_range(x) for x in df[c].astype(str)))
        df[f"{c}_min"] = mins
        df[f"{c}_max"] = maxs
        df[f"{c}_mean"] = means

    return df


def rename_attributes_with_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    colmap = {}
    for c in df.columns:
        if c.endswith(("_min", "_max", "_mean")):
            base = c.rsplit("_", 1)[0]
            target_base = mapping.get(base, base)
            colmap[c] = f"{target_base}_{c.rsplit('_',1)[1]}"
    return df.rename(columns=colmap)


def compute_role_score(df: pd.DataFrame, role: str, roles_data: Dict[str, Dict[str, Dict[str, float]]], mapping: Dict[str, str]) -> pd.DataFrame:
    if role not in roles_data:
        raise KeyError(f"Role '{role}' not found in roles_data")

    weights_nested = roles_data[role]
    weights = {k: v for cat in weights_nested.values() for k, v in cat.items()}

    resolved_cols = {}
    for k, w in weights.items():
        full = mapping.get(k, k)
        candidates = [f"{full}_mean", f"{k}_mean"]
        col = next((c for c in candidates if c in df.columns), None)
        if col:
            resolved_cols[col] = float(w)

    if not resolved_cols:
        raise ValueError(f"No attribute columns for role '{role}' matched the dataset.")

    total_w = sum(resolved_cols.values()) or 1.0
    score = None
    for col, w in resolved_cols.items():
        term = (df[col].astype(float) * w) / total_w
        score = term if score is None else (score + term)
    df = df.copy()
    df["RoleScore"] = score
    return df

###############################################
# Tkinter UI
###############################################

class FMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FM Role Explorer")
        self.geometry("1200x800")

        self.df_raw: pd.DataFrame | None = None
        self.df_view: pd.DataFrame | None = None

        self._build_header()
        self._build_table()
        self._build_plot()

    def _build_header(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(frm, text="Role:").pack(side=tk.LEFT)
        self.role_var = tk.StringVar()
        self.role_combo = ttk.Combobox(frm, textvariable=self.role_var, state="readonly", width=40)
        self.role_combo.pack(side=tk.LEFT, padx=8)

        ttk.Button(frm, text="Open stats.html…", command=self.on_open_html).pack(side=tk.LEFT, padx=4)
        ttk.Button(frm, text="Apply Role & Sort", command=self.on_apply_role).pack(side=tk.LEFT, padx=4)
        ttk.Label(frm, text="(Select up to 3 players in the table to compare below)").pack(side=tk.LEFT, padx=16)

        if roles_data:
            vals = sorted(list(roles_data.keys()))
            self.role_combo["values"] = vals
            if vals:
                self.role_var.set(vals[0])

    def _build_table(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        cols = ("Name", "Position", "Nationality", "Age", "ValueMin_GBP", "ValueMax_GBP", "RoleScore")
        self.tree = ttk.Treeview(container, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, stretch=True, width=120)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_selection_changed)

    def _build_plot(self):
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=4)

        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Selected players vs role attributes (mean)")
        self.ax.set_xlabel("Attribute")
        self.ax.set_ylabel("Mean")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_open_html(self):
        p = filedialog.askopenfilename(title="Open stats.html", filetypes=[("HTML", "*.html"), ("All Files", "*.*")])
        if not p:
            return
        try:
            df = load_stats_html(Path(p))
            df = rename_attributes_with_mapping(df, attribute_mapping)
            self.df_raw = df
            self.df_view = df.copy()
            self.populate_table(self.df_view)
            messagebox.showinfo("Loaded", f"Loaded {len(df)} rows from stats HTML.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load stats: {e}")

    def on_apply_role(self):
        if self.df_raw is None:
            messagebox.showwarning("No data", "Load stats.html first.")
            return
        if not roles_data:
            messagebox.showwarning("No roles", "roles_data.py not found.")
            return
        role = self.role_var.get()
        try:
            df_scored = compute_role_score(self.df_raw, role, roles_data, attribute_mapping)
            df_scored = df_scored.sort_values("RoleScore", ascending=False)
            self.df_view = df_scored
            self.populate_table(self.df_view)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to score role: {e}")

    def populate_table(self, df: pd.DataFrame):
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        display_cols = [c for c in ("Name", "Position", "Nationality", "Age", "ValueMin_GBP", "ValueMax_GBP", "RoleScore") if c in df.columns]
        self.tree["columns"] = display_cols
        for c in display_cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140 if c == "Name" else 110, anchor=tk.W)

        for i, row in df.reset_index(drop=True).iterrows():
            values = [row.get(c, "") for c in display_cols]
            self.tree.insert("", tk.END, iid=str(i), values=values)

    def on_selection_changed(self, event=None):
        if self.df_view is None:
            return
        sel = self.tree.selection()
        if not sel:
            self.ax.clear()
            self.ax.set_title("Selected players vs role attributes (mean)")
            self.ax.set_xlabel("Attribute")
            self.ax.set_ylabel("Mean")
            self.canvas.draw()
            return
        if len(sel) > 3:
            messagebox.showinfo("Limit", "Please select up to 3 players.")
            return

        role = self.role_var.get()
        if not role or not roles_data or not attribute_mapping:
            return
        weights_nested = roles_data.get(role, {})
        weights = {k: v for cat in weights_nested.values() for k, v in cat.items()}

        attr_full_names = [attribute_mapping.get(k, k) for k in weights.keys()]
        mean_cols = [f"{a}_mean" for a in attr_full_names]
        present = [(a, c) for a, c in zip(attr_full_names, mean_cols) if c in self.df_view.columns]
        if not present:
            return

        df_idx = [int(iid) for iid in sel]
        df_sel = self.df_view.iloc[df_idx]

        self.ax.clear()
        attrs = [a for a, _ in present]
        x = np.arange(len(attrs))
        width = 0.8 / max(1, len(df_sel))

        for j, (_, r) in enumerate(df_sel.iterrows()):
            y = [float(r[c]) for _, c in present]
            self.ax.bar(x + j * width, y, width=width, label=str(r.get("Name", f"Player {j+1}")))

        self.ax.set_xticks(x + width * (len(df_sel) - 1) / 2)
        self.ax.set_xticklabels(attrs, rotation=45, ha="right")
        self.ax.set_ylabel("Mean")
        self.ax.set_title(f"{role}: attribute means for selected players")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = FMApp()
    app.mainloop()