from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd
import plotly.express as px
import yaml
from plotly.subplots import make_subplots
import math

from plotting.plotly_dashboard_functions import (
    FUNCTION_BUILDERS,
    apply_common_cleaning,
    normalize_function_figs,
    build_function_figs,
    update_missing_colors,
)

# Mapping of short DB column codes to verbose labels.
INDEX_NAME_MAP_REV = {
    "r": "REGION",
    "l": "NODE",
    "t": "TECHNOLOGY",
    "f": "FUEL",
    "e": "EMISSION",
    "s": "STORAGE",
    "m": "MODE_OF_OPERATION",
    "ts": "TIMESLICE",
    "y": "YEAR",
}

SKIP_TABLES = {"Version", "VariableCost"}
MAPPING_PATH = Path("config/plotting_config_and_timeslices.xlsx")
MISSING_COLOR_CSV = Path("config/missing_plot_colors.csv")
DEFAULT_NO_COLUMNS = None  # None -> auto square; int -> fixed columns
DEFAULT_FUNCTION_OPTIONS = {"drop_zero_categories": True}


def load_plotly_config_yaml(path: Path) -> dict:
    """Load dict/function plot config and layout from YAML if present; inject available_functions."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        print(f"Failed to load plotly config YAML '{path}': {exc}")
        return {}

    # Keep available functions synced from FUNCTION_BUILDERS
    available = sorted(FUNCTION_BUILDERS.keys())
    if data.get("available_functions") != available:
        data["available_functions"] = available
        try:
            yaml.safe_dump(data, path.open("w"), sort_keys=False)
        except Exception:
            pass
    else:
        data["available_functions"] = available
    return data


def _list_result_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'v%' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    return [row[0] for row in cur.fetchall() if row[0] not in SKIP_TABLES]


def _load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    df = df.drop(columns=[c for c in df.columns if c.lower() == "solvedtm"], errors="ignore")
    rename_map = {col: INDEX_NAME_MAP_REV.get(col, col) for col in df.columns}
    if "val" in df.columns:
        rename_map["val"] = "VALUE"
    if "y" in df.columns:
        rename_map["y"] = "YEAR"
    df = df.rename(columns=rename_map)
    if "YEAR" in df.columns:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    return df


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df.copy()
    for col, spec in filters.items():
        if col not in out.columns:
            continue
        if isinstance(spec, (str, int, float)):
            out = out[out[col] == spec]
            continue
        if isinstance(spec, dict):
            if any(k in spec for k in ("min", "max")):
                # Try to coerce to numeric to avoid string/int comparisons
                out[col] = pd.to_numeric(out[col], errors="coerce")
                out = out.dropna(subset=[col])
            if "eq" in spec:
                out = out[out[col] == spec["eq"]]
            if "in" in spec:
                out = out[out[col].isin(spec["in"])]
            if "min" in spec:
                out = out[out[col] >= spec["min"]]
            if "max" in spec:
                out = out[out[col] <= spec["max"]]
    return out


def _transform_df(df: pd.DataFrame, cfg: dict, required_cols: set[str]) -> pd.DataFrame:
    transform = (cfg.get("transform") or "none").lower()
    groupby = cfg.get("groupby")
    agg = cfg.get("agg", "sum")
    if transform == "none":
        return df

    if transform in {"sum_by_year", "mean_by_year"}:
        # If requested columns (e.g., color) would be dropped, skip the transform.
        if any(col not in {"YEAR", "VALUE"} for col in required_cols if col):
            return df
        if "YEAR" not in df.columns or "VALUE" not in df.columns:
            return df
        agg_func = "sum" if "sum" in transform else "mean"
        return df.groupby(["YEAR"], dropna=False)["VALUE"].agg(agg_func).reset_index()

    if transform in {"sum_by_group", "mean_by_group"}:
        if "VALUE" not in df.columns:
            return df
        group_cols = list(groupby) if groupby else []
        agg_func = "sum" if "sum" in transform else "mean"
        if "YEAR" in df.columns and "YEAR" not in group_cols:
            group_cols.append("YEAR")
        # If required columns are outside group_cols+VALUE, skip transform to preserve them.
        allowed_cols = set(group_cols) | {"VALUE", "YEAR"}
        if any(col and col not in allowed_cols for col in required_cols):
            return df
        if not group_cols:
            return df
        return df.groupby(group_cols, dropna=False)["VALUE"].agg(agg_func).reset_index()

    if transform == "groupby":
        if not groupby or "VALUE" not in df.columns:
            return df
        if any(col and col not in set(groupby) | {"VALUE", "YEAR"} for col in required_cols):
            return df
        return df.groupby(groupby, dropna=False)["VALUE"].agg(agg).reset_index()

    return df


def _build_fig(df: pd.DataFrame, cfg: dict):
    plot_type = (cfg.get("plot") or "line").lower()
    x = cfg.get("x")
    y = cfg.get("y", "VALUE")
    color = cfg.get("color")
    facet_row = cfg.get("facet_row")
    facet_col = cfg.get("facet_col")
    labels = cfg.get("labels")
    title = cfg.get("title") or cfg.get("table")
    hover_data = cfg.get("hover_data")

    color_map = cfg.get("color_map")

    if plot_type == "bar":
        fig = px.bar(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, labels=labels, hover_data=hover_data, title=title, color_discrete_map=color_map)
    elif plot_type == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, labels=labels, hover_data=hover_data, title=title, color_discrete_map=color_map)
    elif plot_type == "area":
        fig = px.area(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, labels=labels, hover_data=hover_data, title=title, color_discrete_map=color_map)
    else:
        fig = px.line(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, labels=labels, hover_data=hover_data, title=title, color_discrete_map=color_map)
    return fig


def generate_plotly_dashboard(
    db_path: Path,
    output_path: Path,
    plots_config_dict: Iterable[dict] | None,
    layout: str = "scroll",  # "scroll" (stacked cards) or "grid" (subplot figure)
    function_figs: list[str] | None = None,
    config_yaml: Path | None = None,
    no_columns: int | None = None,
) -> Path | None:
    """
    Build an HTML page with a set of plotly express figures based on a simple config list.
    Each config dict can specify:
      - table (required): result table name
      - plot: line|bar|scatter|area (default line)
      - x, y, color, facet_row, facet_col, title, labels, hover_data
      - filters: {COL: value or {eq/in/min/max}}
      - transform: none|sum_by_year|mean_by_year|sum_by_group|mean_by_group|groupby
      - groupby: list of columns (used for transform sum/mean/groupby)
      - agg: aggregator name when transform=groupby (default sum)
    """
    # proceed even if no dict configs; function-based figs may still apply

    db_path = Path(db_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    breakpoint()
    yaml_cfg = load_plotly_config_yaml(config_yaml) if config_yaml else {}
    plots_cfg = plots_config_dict or yaml_cfg.get("dict_figs")
    fn_figs_cfg = function_figs if function_figs is not None else yaml_cfg.get("function_figs")
    fn_figs = normalize_function_figs(fn_figs_cfg)
    no_columns = no_columns if no_columns is not None else yaml_cfg.get("no_columns", DEFAULT_NO_COLUMNS)
    layout = (yaml_cfg.get("layout") or layout or "scroll").strip().lower()

    conn = sqlite3.connect(db_path)
    available_tables = set(_list_result_tables(conn))

    mapping = _load_mapping()
    missing_colors: set[str] = set()
    figs: list = []
    for cfg in plots_cfg or []:
        table = cfg.get("table")
        if not table:
            print("  Skipping plot config with no table.")
            continue
        if table not in available_tables:
            print(f"  Skipping plot '{cfg.get('title', table)}': table '{table}' not found or not allowed.")
            continue

        df = _load_table(conn, table)
        if df.empty:
            print(f"  Skipping plot '{table}': empty table.")
            continue

        df = _apply_filters(df, cfg.get("filters"))
        cfg = dict(cfg)
        # If user didn't specify x/y, set simple defaults.
        if "x" not in cfg and "YEAR" in df.columns:
            cfg["x"] = "YEAR"
        if "y" not in cfg and "VALUE" in df.columns:
            cfg["y"] = "VALUE"

        required_cols: set[str] = set()
        for key in ("x", "y", "color", "facet_row", "facet_col"):
            val = cfg.get(key)
            if isinstance(val, str):
                required_cols.add(val)
        hover = cfg.get("hover_data")
        if isinstance(hover, (list, tuple)):
            required_cols.update([h for h in hover if isinstance(h, str)])

        # Apply mapping-based color map when color column matches mapping keys.
        color_map = None
        color_col = cfg.get("color")
        # Optional category cleaning for dict plots
        func_opts = cfg.get("function_options") or {}
        category_col = cfg.get("category_col") or color_col
        if category_col and category_col in df.columns:
            df = apply_common_cleaning(df, category_col, func_opts)

        df = _transform_df(df, cfg, required_cols)

        if color_col and color_col in df.columns:
            # Prefer tech colors; fall back to fuel colors if color col looks like FUEL.
            color_map = mapping.get("colors", {})
        cfg = dict(cfg)
        cfg["color_map"] = color_map
        if color_col and color_map:
            update_missing_colors(df[color_col].dropna().unique(), color_map, missing_colors)

        try:
            fig = _build_fig(df, cfg)
            fig.update_layout(
                height=420,
                autosize=True,
                margin=dict(l=40, r=180, t=80, b=60),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=10),
                    itemsizing="trace",
                    tracegroupgap=4,
                    bgcolor="rgba(255,255,255,0.7)",
                ),
                title=dict(x=0.02, y=0.94, font=dict(size=18)),
            )
            figs.append(fig)
        except Exception as exc:
            breakpoint()
            print(f"  Failed to build plot for '{table}': {exc}")
            continue

    conn.close()

    # Always allow function-based figs; function_figs None -> all; [] -> none.
    try:
        figs.extend(build_function_figs(db_path, mapping, include=fn_figs, missing_colors=missing_colors))
    except Exception as exc:
        breakpoint()
        print(f"Failed to build function-based plots: {exc}")

    if not figs:
        breakpoint()
        print("No dashboards generated.")
        return None

    _persist_missing_colors(missing_colors, mapping.get("colors", {}))

    create_dashboard_html(figs=figs, output_path=output_path, layout=layout, no_columns=no_columns)
    return output_path


def _grid_cols(count: int, no_columns: int | None = None) -> int:
    best_rows, best_cols = count, 1
    if no_columns:
        return max(1, min(count, int(no_columns)))
    for cols in range(1, count + 1):
        rows = -(-count // cols)  # ceil division
        if rows < cols:
            continue
        score = (rows - cols, rows * cols)
        best_score = (best_rows - best_cols, best_rows * best_cols)
        if score < best_score:
            best_rows, best_cols = rows, cols
    return best_cols


def create_dashboard_html(figs: list, output_path: Path, layout: str = "scroll", no_columns: int | None = None):
    """Create an HTML dashboard from plotly figures.

    layout="scroll" -> CSS grid of individual HTML fragments (scrollable if many).
    layout="grid"   -> single Plotly figure with subplots (no scrolling; legends de-duped, placed right).
    """
    grid_cols = _grid_cols(len(figs), no_columns=no_columns)

    if layout == "grid":
        # Legacy-style grid: single figure with subplots; no scrolling expected.
        if no_columns:
            cols = max(1, min(int(no_columns), len(figs)))
            rows = int(math.ceil(len(figs) / cols))
        else:
            rows = int(math.ceil(math.sqrt(len(figs))))
            cols = int(math.ceil(len(figs) / rows))
        subplot_titles = [getattr(f.layout.title, "text", "") or "" for f in figs]
        combo = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "xy"} for _ in range(cols)] for _ in range(rows)],
        )
        for i, f in enumerate(figs):
            r = i // cols + 1
            c = i % cols + 1
            for tr in f.data:
                combo.add_trace(tr, row=r, col=c)
        # Remove duplicate legend entries.
        names = set()
        combo.for_each_trace(
            lambda tr: tr.update(showlegend=False) if (tr.name in names) else names.add(tr.name)
        )
        combo.update_layout(
            title_text="NEMO Results Dashboard",
            legend=dict(
                orientation="v",
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=10),
            ),
            autosize=True,
            margin=dict(l=40, r=200, t=80, b=50),
        )
        # Let Plotly determine sizing; no manual height forcing.
        combo.write_html(output_path, include_plotlyjs="cdn", full_html=True)
        print(f"Wrote plotly dashboard with {len(figs)} plot(s) to '{output_path}' (subplot grid).")
        return

    style = (
        "<style>"
        f".grid {{display:grid; grid-template-columns:repeat({grid_cols}, 1fr); gap:16px;}}"
        ".card {background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,0.05);}"
        ".card .plotly-graph-div {width:100%!important; height:420px!important;}"
        "body {font-family:Arial, sans-serif; margin:20px; background:#f7f7f9;}"
        "h1 {margin-bottom:16px; font-size:28px;}"
        "@media (max-width: 640px) {.grid {grid-template-columns:1fr;}}"
        "</style>"
    )
    html_blocks: list[str] = []
    for fig in figs:
        html_blocks.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                default_width="100%",
                default_height="100%",
            )
        )

    html = [
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>NEMO Results Dashboard</title>",
        style,
        "</head><body>",
        "<h1>NEMO Results Dashboard</h1>",
        "<div class='grid'>",
    ]
    for block in html_blocks:
        html.append(f"<div class='card'>{block}</div>")
    html.append("</div></body></html>")
    output_path.write_text("\n".join(html), encoding="utf-8")
    print(f"Wrote plotly dashboard with {len(figs)} plot(s) to '{output_path}'.")


def put_all_graphs_in_one_html(figs: list, output_path: Path):
    """Basic aggregation: stack all figures into a single HTML file without layout tweaks."""
    body_parts = []
    for fig in figs:
        inner_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        body_parts.append(inner_html)
    html = "<html><head><meta charset='utf-8'></head><body>" + "\n".join(body_parts) + "</body></html>"
    output_path.write_text(html, encoding="utf-8")


def _load_mapping() -> dict:
    if not MAPPING_PATH.exists():
        return {"colors": {}, "powerplant": None, "fuel": None}
    try:
        mapping = pd.read_excel(MAPPING_PATH, sheet_name=None)
    except Exception:
        return {"colors": {}, "powerplant": None, "fuel": None}
    colors = {}
    if "plotting_name_to_color" in mapping:
        try:
            colors = mapping["plotting_name_to_color"].set_index("plotting_name").to_dict().get("color", {})
        except Exception:
            colors = {}
    return {
        "colors": colors,
        "powerplant": mapping.get("POWERPLANT"),
        "fuel": mapping.get("FUEL"),
    }


def _load_missing_colors() -> set[str]:
    if not MISSING_COLOR_CSV.exists():
        return set()
    try:
        df = pd.read_csv(MISSING_COLOR_CSV)
        return set(df["label"].astype(str).tolist())
    except Exception:
        return set()


def _persist_missing_colors(missing: set[str], color_map: dict):
    """Update missing color CSV by removing resolved labels and adding new ones."""
    existing = _load_missing_colors()
    known = {str(k) for k in (color_map or {}).keys()}
    updated = (existing | missing) - known
    if not updated:
        if MISSING_COLOR_CSV.exists():
            MISSING_COLOR_CSV.unlink(missing_ok=True)
        return
    df = pd.DataFrame(sorted(updated), columns=["label"])
    MISSING_COLOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MISSING_COLOR_CSV, index=False)
