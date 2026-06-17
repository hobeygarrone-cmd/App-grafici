"""
GUI Grafici — versione web (Streamlit)
Deployment: streamlit run streamlit_app.py
             oppure Streamlit Cloud (share.streamlit.io)
"""

import io
import os
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import pandas as pd
import streamlit as st

# ── Percorsi ──────────────────────────────────────────────────────────────────
_DIR  = pathlib.Path(__file__).parent
_XGUI = _DIR / "xGUI"

# ── Geopandas (opzionale: solo per mappe) ─────────────────────────────────────
try:
    import geopandas as gpd
    GEO_OK = True
except ImportError:
    GEO_OK = False

# ── Configurazione pagina ──────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="GUI Grafici", page_icon="📊")
st.title("📊 GUI Grafici")

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {"df": None, "col_roles": {}, "fig": None}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Costanti ───────────────────────────────────────────────────────────────────
CHART_TYPES = [
    ("Barre",              "bar"),
    ("Barre impilate",     "bar_stacked"),
    ("Linee",              "line"),
    ("Dispersione (XY)",   "scatter"),
    ("Istogramma",         "histogram"),
    ("Torta",              "pie"),
    ("Heatmap",            "heatmap"),
    ("Boxplot",            "boxplot"),
    ("Radar",              "radar"),
    ("Imbuto (Funnel)",    "funnel"),
    ("Podio",              "podio"),
    ("Violino",            "violin"),
    ("Barre + Linee",      "combo"),
    ("Mappa coropletica",  "choropleth"),
    ("Mappa bolla",        "bubble_map"),
    ("Mappa bande",        "band_map"),
    ("Mappa pin",          "pin_map"),
]

MAP_KEYS    = {"choropleth", "bubble_map", "band_map", "pin_map"}
# Grafici che usano assi personalizzati (axis off / polare) — _apply_axis_settings saltata
_NO_AXIS_KEYS = {"funnel", "podio"}
_FREQ_LABEL   = "— Frequenza —"

_SDG_COLORS = [
    "#E5243B","#DDA63A","#4C9F38","#C5192D","#FF3A21",
    "#26BDE2","#FCC30B","#A21942","#FD6925","#DD1367",
    "#FD9D24","#BF8B2E","#3F7E44","#0A97D9","#56C02B",
    "#00689D","#19486A",
]

# Palette personalizzate (nomi anonimi basati sui colori dominanti)
_BRAND_PALETTES = {
    "Viola e Oro":       ["#312A74","#FDC400","#5C5499","#F0A800","#1E1A55","#FFD966","#7B73AA","#E8C000"],
    "Petrolio e Rosso":  ["#004258","#B10931","#326779","#D43060","#005A74","#E05070","#007A9A","#FF6080"],
    "Navy e Rosso 1":    ["#002F6C","#E31937","#0057B8","#FF4D4D","#003F8A","#C0392B","#1565C0","#FF7043"],
    "Antracite e Rosso": ["#1C1C1C","#D40000","#555555","#FF6666","#333333","#B30000","#888888","#FF9999"],
    "Navy e Teal":       ["#1E3264","#3D4FA0","#9089C0","#2A6B5A","#3B9878","#7BC5A0","#F0A020","#888FA0"],
    "Navy e Ambra":      ["#21345C","#5B8DC8","#3EA8A5","#D98A25","#8870B0","#89C0A5","#A89880","#C0CEDE"],
    "Navy e Oro":        ["#1C2B5A","#3C68B5","#E8A020","#8890A0","#5878C8","#C47A10","#2A3870","#B0B8C8"],
    "Blu e Arancio":     ["#156082","#E97132","#196B24","#0F9ED5","#A02B93","#4EA72E","#467886","#96607D"],
    "SDG":               _SDG_COLORS,
}

PALETTES = {
    # ── Personalizzate ───────────────────────────────────────────────────────
    "Viola e Oro":       "Viola e Oro",
    "Petrolio e Rosso":  "Petrolio e Rosso",
    "Navy e Rosso 1":    "Navy e Rosso 1",
    "Antracite e Rosso": "Antracite e Rosso",
    "Navy e Teal":       "Navy e Teal",
    "Navy e Ambra":      "Navy e Ambra",
    "Navy e Oro":        "Navy e Oro",
    "Blu e Arancio":     "Blu e Arancio",
    "SDG":               "SDG",
    # ── Qualitative matplotlib ───────────────────────────────────────────────
    "Tab10":     "tab10",
    "Tab20":     "tab20",
    "Set1":      "Set1",
    "Set2":      "Set2",
    "Set3":      "Set3",
    "Paired":    "Paired",
    "Pastel":    "Pastel1",
    "Scuro":     "Dark2",
    # ── Sequenziali / divergenti ─────────────────────────────────────────────
    "Blues":     "Blues",
    "Greens":    "Greens",
    "Reds":      "Reds",
    "Oranges":   "Oranges",
    "Purples":   "Purples",
    "YlOrRd":    "YlOrRd",
    "RdYlGn":    "RdYlGn",
    "Viridis":   "viridis",
    "Plasma":    "plasma",
    "Coolwarm":  "coolwarm",
}

MAP_SCOPES = [
    ("Piemonte — Province",               "prov_piemonte"),
    ("Piemonte + Val d'Aosta — Province", "prov_piemvda"),
    ("Italia — Province",                 "prov_italy"),
    ("Italia — Regioni",                  "reg_italy"),
    ("Europa — Paesi",                    "europe"),
    ("Piemonte — Comuni",                 "comuni_piemonte"),
    ("Val d'Aosta — Comuni",              "comuni_valdaosta"),
    # CAP: usa file pre-filtrati generati da prep_geo.py (non cap_ita.geojson che è 206 MB)
    ("Piemonte + Val d'Aosta — CAP",       "cap_piemvda"),
    ("Città Metropolitana di Torino — CAP", "cap_torino"),
    ("Torino — CAP",                        "cap_torino_city"),
    ("Torino — Circoscrizioni",             "circ_torino"),
]
MAP_SCOPE_KEYS = {lbl: key for lbl, key in MAP_SCOPES}

_PIEMONTE_PROV = {
    "Torino", "Cuneo", "Alessandria", "Asti", "Novara",
    "Vercelli", "Biella", "Verbano-Cusio-Ossola",
}
_PIEMVDA_PROV = _PIEMONTE_PROV | {"Valle d'Aosta/Vallée d'Aoste", "Aosta"}
_CAP_PREFIXES_PIEMVDA = {"10", "11", "12", "13", "14", "15", "28"}
_CAP_PREFIXES_TORINO  = {"10"}
_CAP_TORINO_CITY = (
    {"10100"}
    | {str(i) for i in range(10121, 10132)}
    | {str(i) for i in range(10132, 10138)}
    | {str(i) for i in range(10138, 10142)}
    | {str(i) for i in range(10142, 10149)}
    | {str(i) for i in range(10151, 10157)}
)

# ── Caricamento geometrie (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Caricamento geometrie mappa…")
def _load_geo(scope_key: str):
    """Carica il GeoDataFrame per l'ambito scelto. Ritorna None se non disponibile."""
    if not GEO_OK:
        return None

    _PROV_URL  = ("https://raw.githubusercontent.com/openpolis/geojson-italy"
                  "/master/geojson/limits_IT_provinces.geojson")
    _REG_URL   = ("https://raw.githubusercontent.com/openpolis/geojson-italy"
                  "/master/geojson/limits_IT_regions.geojson")
    _PROV_LOC  = _XGUI / "province_italiane.geojson"
    _REG_LOC   = _XGUI / "regioni_italiane.geojson"
    _PIE_COM   = _XGUI / "ondata_confini_amministrativi_api_v2_it_20250101_piemonte_comuni.geo.json"
    _VDA_COM   = _XGUI / "ondata_confini_amministrativi_api_v2_it_20250101_valdaosta_comuni.geo.json"
    _CAP_FILE  = _XGUI / "cap_ita.geojson"
    _EUR_FILE  = _XGUI / "europe.geojson"

    def _name_col(gdf, candidates):
        return next((c for c in candidates if c in gdf.columns), gdf.columns[0])

    try:
        if scope_key in ("prov_piemonte", "prov_piemvda", "prov_italy"):
            src = str(_PROV_LOC) if _PROV_LOC.exists() else _PROV_URL
            gdf = gpd.read_file(src)
            nc  = _name_col(gdf, ["prov_name", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            if scope_key == "prov_piemonte":
                gdf = gdf[gdf["GEO_NAME"].isin(_PIEMONTE_PROV)].copy()
            elif scope_key == "prov_piemvda":
                gdf = gdf[gdf["GEO_NAME"].isin(_PIEMVDA_PROV)].copy()
            return gdf.to_crs(epsg=32632)

        elif scope_key == "reg_italy":
            src = str(_REG_LOC) if _REG_LOC.exists() else _REG_URL
            gdf = gpd.read_file(src)
            nc  = _name_col(gdf, ["reg_name", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            return gdf.to_crs(epsg=32632)

        elif scope_key == "europe":
            if not _EUR_FILE.exists():
                st.error("europe.geojson non trovato nella cartella xGUI/")
                return None
            gdf = gpd.read_file(str(_EUR_FILE))
            nc  = _name_col(gdf, ["name", "NAME", "SOVEREIGNT", "sovereignt",
                                   "NAME_LONG", "name_long", "admin", "ADMIN",
                                   "COUNTRY", "country"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            gdf["GEO_NAME"] = gdf["GEO_NAME"].astype(str).str.strip()
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=3035)

        elif scope_key in ("comuni_piemonte", "comuni_valdaosta"):
            files = []
            if scope_key in ("comuni_piemonte",) and _PIE_COM.exists():
                files.append(str(_PIE_COM))
            if scope_key in ("comuni_valdaosta",) and _VDA_COM.exists():
                files.append(str(_VDA_COM))
            if not files:
                st.error("File comuni non trovato in xGUI/"); return None
            gdfs = [gpd.read_file(f) for f in files]
            gdf  = pd.concat(gdfs, ignore_index=True) if len(gdfs) > 1 else gdfs[0]
            gdf  = gpd.GeoDataFrame(gdf, geometry="geometry")
            nc   = _name_col(gdf, ["name", "comune", "COMUNE",
                                    "denominazione_ita", "DENOMINAZI", "NAME", "nome"])
            gdf  = gdf.rename(columns={nc: "GEO_NAME"})
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=32632)

        elif scope_key in ("cap_piemvda", "cap_torino", "cap_torino_city"):
            # Usa file pre-filtrati generati da prep_geo.py
            _cap_files = {
                "cap_piemvda":     _XGUI / "cap_piemvda.geojson",
                "cap_torino":      _XGUI / "cap_torino.geojson",
                "cap_torino_city": _XGUI / "cap_torino_city.geojson",
            }
            _f = _cap_files[scope_key]
            if not _f.exists():
                st.error(
                    f"{_f.name} non trovato in xGUI/. "
                    "Esegui prima `python prep_geo.py` per generare i file CAP filtrati."
                )
                return None
            gdf = gpd.read_file(str(_f))
            nc  = _name_col(gdf, ["cap", "CAP", "GEO_NAME", "codice_postale",
                                   "zip", "postalcode", "name", "NAME"])
            gdf = gdf.rename(columns={nc: "GEO_NAME"})
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf.to_crs(epsg=32632)

        elif scope_key == "circ_torino":
            _CIRC_FILE = _XGUI / "circoscrizioni.csv"
            if not _CIRC_FILE.exists():
                st.error("circoscrizioni.csv non trovato in xGUI/"); return None
            try:
                import csv as _csv
                from shapely import wkt as _swkt
                rows = []
                with open(str(_CIRC_FILE), encoding="utf-8", newline="") as f:
                    for r in _csv.DictReader(f, delimiter=";"):
                        rows.append(r)
                df_c  = pd.DataFrame(rows)
                geoms = []
                for w in df_c["WKT_GEOM"]:
                    try:    geoms.append(_swkt.loads(w))
                    except: geoms.append(None)
                gdf = gpd.GeoDataFrame(df_c, geometry=geoms, crs="EPSG:3003")
                gdf["DENOM"] = gdf["DENOM"].astype(str).str.strip()
                gdf = gdf.rename(columns={"DENOM": "GEO_NAME"})
                return gdf.to_crs(epsg=32632)
            except Exception as e:
                st.error(f"Errore circoscrizioni: {e}"); return None

    except Exception as e:
        st.error(f"Errore caricamento geometrie: {e}")
    return None


# ── Helper generici ────────────────────────────────────────────────────────────
def _is_num(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().mean() > 0.5


def _smart_labels(ax_, labels, override_rot=None):
    """Etichette X con rotazione/wrapping automatici."""
    strs = [str(l) for l in labels]
    n = len(strs)
    if n == 0:
        return
    if override_rot is not None and override_rot >= 0:
        ha = "right" if override_rot > 0 else "center"
        ax_.set_xticklabels(strs, rotation=override_rot, ha=ha, fontsize=8, rotation_mode="anchor")
        return
    max_len = max(len(s) for s in strs)
    if n > 25:
        step = max(1, round(n / 15))
        sparse = [s if i % step == 0 else "" for i, s in enumerate(strs)]
        ax_.set_xticklabels(sparse, rotation=70, ha="right", fontsize=6, rotation_mode="anchor")
    elif n > 10 or max_len > 8:
        rot = 70 if max_len > 12 else 45
        ax_.set_xticklabels(strs, rotation=rot, ha="right", fontsize=7, rotation_mode="anchor")
    elif max_len > 10:
        wrapped = ["\n".join(textwrap.wrap(s, 12)) for s in strs]
        ax_.set_xticklabels(wrapped, rotation=0, ha="center", fontsize=8)
        ax_.tick_params(axis="x", pad=4)
    else:
        ax_.set_xticklabels(strs, rotation=30, ha="right", fontsize=8, rotation_mode="anchor")


def _xkey(v):
    """Chiave di ordinamento: numerica se possibile, altrimenti stringa."""
    try:
        return (0, float(v))
    except (ValueError, TypeError):
        return (1, str(v))


def _sort_df_by_x(df_, x_col):
    """Ordina df_ per x_col: numericamente se tutti i valori sono numerici, altrimenti come stringa."""
    nums = pd.to_numeric(df_[x_col], errors="coerce")
    if nums.notna().all():
        return df_.sort_values(x_col, key=lambda s: pd.to_numeric(s, errors="coerce"))
    return df_.sort_values(x_col, key=lambda s: s.astype(str))


def _agg(df_, x_col, y_col, grp_col=None):
    g_cols = [c for c in [x_col, grp_col] if c]
    if y_col and g_cols:
        return df_.groupby(g_cols, sort=False)[y_col].sum().reset_index()
    if x_col:
        return df_[x_col].value_counts().reset_index().rename(
            columns={x_col: x_col, "count": "_n"})
    return df_


def _agg_by_x(df_, x_cols, y_col):
    """Aggrega df_ per x_cols: frequenza se y_col == _FREQ_LABEL, altrimenti somma numerica.
    x_cols può essere str (singolo) o list (heatmap: [col_cols, row_col]).
    Restituisce df ordinato per il primo x_col, con colonne [*x_cols, value_col]."""
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    if y_col == _FREQ_LABEL or not y_col:
        out = df_.groupby(x_cols, sort=False).size().reset_index(name="Frequenza")
        val_col = "Frequenza"
    else:
        out = df_.groupby(x_cols, sort=False)[y_col].sum().reset_index()
        val_col = y_col
    # ordina sempre per il primo asse X (numerico se possibile)
    out = _sort_df_by_x(out, x_cols[0])
    return out, val_col


def _colors(n, palette_name):
    brand = _BRAND_PALETTES.get(palette_name)
    if brand:
        return [brand[i % len(brand)] for i in range(n)]
    try:
        cmap = plt.get_cmap(palette_name) if palette_name else plt.get_cmap("tab10")
    except Exception:
        cmap = plt.get_cmap("tab10")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _get_cmap(palette_name, fallback="Blues"):
    """Restituisce un matplotlib colormap; per palette brand usa ListedColormap."""
    brand = _BRAND_PALETTES.get(palette_name)
    if brand:
        from matplotlib.colors import ListedColormap
        return ListedColormap(brand)
    try:
        return plt.get_cmap(palette_name) if palette_name else plt.get_cmap(fallback)
    except Exception:
        return plt.get_cmap(fallback)


def _cat_colors(categories, palette_name):
    cats = list(categories)
    cols = _colors(len(cats), palette_name)
    return dict(zip(cats, cols))


def _apply_style(ax_, title, xlabel, ylabel, show_legend, show_grid, opts=None):
    o = opts or {}
    title_sz   = o.get("title_sz", 13)
    title_bold = o.get("title_bold", True)
    title_ital = o.get("title_ital", False)
    label_sz   = o.get("label_sz", 10)
    minimal    = o.get("minimal", False)

    if title:
        ax_.set_title(
            title,
            fontsize=title_sz,
            fontweight="bold" if title_bold else "normal",
            fontstyle="italic" if title_ital else "normal",
            pad=10,
        )
    if xlabel:
        ax_.set_xlabel(xlabel, fontsize=label_sz)
    if ylabel:
        ax_.set_ylabel(ylabel, fontsize=label_sz)
    if show_grid:
        ax_.grid(True, alpha=0.3, linestyle="--", axis="y")
    if show_legend:
        handles, labs = ax_.get_legend_handles_labels()
        if handles:
            ax_.legend(fontsize=o.get("legend_sz", 8), framealpha=0.85)

    # Limiti Y
    y_min_s = o.get("y_min", "")
    y_max_s = o.get("y_max", "")
    if y_min_s or y_max_s:
        cur_lo, cur_hi = ax_.get_ylim()
        try:
            ax_.set_ylim(
                float(y_min_s) if y_min_s else cur_lo,
                float(y_max_s) if y_max_s else cur_hi,
            )
        except ValueError:
            pass

    # Stile minimal: rimuove spine superiore e destra
    if minimal:
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)

    # Mostra / nascondi asse Y
    if not o.get("show_axes_y", True):
        ax_.yaxis.set_visible(False)
        ax_.spines["left"].set_visible(False)

    # Sfondo bianco
    if o.get("white_bg", False):
        ax_.set_facecolor("white")
        if ax_.get_figure():
            ax_.get_figure().patch.set_facecolor("white")


_GRID_STYLE_MAP = {
    "—  solida":        "-",
    "-- tratteggiata":  "--",
    ":  punteggiata":   ":",
    "-. tratto-punto":  "-.",
}
_SCALE_TYPES = ["Normale", "Log10", "Log₂", "Sqrt"]


def _apply_axis_settings(ax, opts):
    """Applica scala, minor tick, griglia avanzata e dimensioni font assi."""
    import matplotlib.ticker as mticker
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter
    from matplotlib.ticker import FixedLocator

    for axis_name in ("x", "y"):
        sc     = str(opts.get(f"{axis_name}_scale", "Normale"))
        step_s = str(opts.get(f"{axis_name}_step", "")).strip()
        set_scale = ax.set_xscale if axis_name == "x" else ax.set_yscale
        mpl_axis  = ax.xaxis     if axis_name == "x" else ax.yaxis
        set_lim   = ax.set_xlim  if axis_name == "x" else ax.set_ylim
        is_log    = sc in ("Log10", "Log₂")
        # Salta scala su assi categoriali (FixedLocator = barre con etichette)
        is_cat = isinstance(mpl_axis.get_major_locator(), FixedLocator)
        if not is_cat:
            try:
                if sc == "Log10":
                    set_scale("log", base=10)
                elif sc == "Log₂":
                    set_scale("log", base=2)
                elif sc == "Sqrt":
                    set_scale("function", functions=(
                        lambda v: np.sqrt(np.clip(v, 0, None)),
                        lambda v: np.square(v),
                    ))
            except Exception:
                pass
            # Scala log: assicura limite inferiore > 0 e formatter leggibile
            if is_log:
                lo, hi = (ax.get_xlim() if axis_name == "x" else ax.get_ylim())
                if lo <= 0 < hi:
                    try:
                        dmin, _ = mpl_axis.get_data_interval()
                        pos_lo  = dmin if dmin > 0 else hi / 100
                    except Exception:
                        pos_lo = hi / 100
                    try:
                        set_lim(pos_lo, hi)
                    except Exception:
                        pass
                mpl_axis.set_major_formatter(
                    mticker.FuncFormatter(
                        lambda v, pos: f"{int(v):,}" if v >= 1 and v == int(v) else f"{v:g}"
                    )
                )
            # Passo tick personalizzato (non su log, che usa LogLocator)
            if step_s and not is_log:
                try:
                    mpl_axis.set_major_locator(MultipleLocator(float(step_s)))
                except Exception:
                    pass
            # Minor tick
            if opts.get(f"{axis_name}_minor", False):
                try:
                    if is_log:
                        # Per scala log: minor tick tra le decade (2,3,...9)
                        mpl_axis.set_minor_locator(LogLocator(subs=list(range(2, 10))))
                        mpl_axis.set_minor_formatter(NullFormatter())
                    else:
                        ndiv = int(opts.get(f"{axis_name}_minor_ndiv", 5))
                        mpl_axis.set_minor_locator(AutoMinorLocator(ndiv))
                except Exception:
                    pass

    # ── Griglia avanzata ──────────────────────────────────────────────────────
    _has_adv_grid = any(opts.get(k, False) for k in
                        ("gx_maj", "gx_min", "gy_maj", "gy_min"))
    if _has_adv_grid:
        # Azzera griglia precedente per tutti gli assi / livelli
        ax.grid(False, which="both", axis="both")

    for axis_name, pfx in [("y", "gy"), ("x", "gx")]:
        if opts.get(f"{pfx}_maj", False):
            sty = _GRID_STYLE_MAP.get(str(opts.get(f"{pfx}_maj_sty", "--")), "--")
            try:
                alp = float(opts.get(f"{pfx}_maj_alp", 0.4))
                lw  = float(opts.get(f"{pfx}_maj_lw",  0.8))
                ax.grid(True, which="major", axis=axis_name,
                        linestyle=sty, alpha=alp, linewidth=lw)
            except Exception:
                pass
        if opts.get(f"{pfx}_min", False):
            sty = _GRID_STYLE_MAP.get(str(opts.get(f"{pfx}_min_sty", ":")), ":")
            try:
                alp  = float(opts.get(f"{pfx}_min_alp", 0.25))
                lw   = float(opts.get(f"{pfx}_min_lw",  0.5))
                mpl_a = ax.xaxis if axis_name == "x" else ax.yaxis
                # Minor locator (se non già impostato dalla sezione minor tick)
                if not opts.get(f"{axis_name}_minor", False):
                    ndiv = int(opts.get(f"{pfx}_min_ndiv", 5))
                    sc_n = str(opts.get(f"{axis_name}_scale", "Normale"))
                    if sc_n in ("Log10", "Log₂"):
                        mpl_a.set_minor_locator(LogLocator(subs=list(range(2, 10))))
                        mpl_a.set_minor_formatter(NullFormatter())
                    else:
                        mpl_a.set_minor_locator(AutoMinorLocator(ndiv))
                ax.grid(True, which="minor", axis=axis_name,
                        linestyle=sty, alpha=alp, linewidth=lw)
            except Exception:
                pass

    # ── Font size etichette tick e legenda ────────────────────────────────────
    xtick_sz  = opts.get("xtick_sz")
    ytick_sz  = opts.get("ytick_sz")
    legend_sz = opts.get("legend_sz")
    if xtick_sz:
        ax.tick_params(axis="x", labelsize=int(xtick_sz))
    if ytick_sz:
        ax.tick_params(axis="y", labelsize=int(ytick_sz))
    if legend_sz:
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(int(legend_sz))


def _normalize_geo(series: pd.Series, name_up: dict) -> pd.Series:
    """Normalizza i nomi geografici: maiuscolo + risolve alias."""
    return series.astype(str).str.strip().str.upper().map(name_up).fillna(
        series.astype(str).str.strip()
    )


def _band_split_v(geom, cum_area, x_lo, x_hi):
    """Ricerca binaria: x tale che area(geom ∩ [x_lo, x]) ≈ cum_area."""
    from shapely.geometry import box as _sb
    miny, maxy = geom.bounds[1], geom.bounds[3]
    lo, hi = x_lo, x_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        a = geom.intersection(_sb(x_lo, miny, mid, maxy)).area
        if abs(a - cum_area) / (abs(cum_area) + 1e-10) < 1e-4:
            return mid
        lo, hi = (mid, hi) if a < cum_area else (lo, mid)
    return (lo + hi) / 2


def _band_split_h(geom, cum_area, y_lo, y_hi):
    """Ricerca binaria: y tale che area(geom ∩ [y_lo, y]) ≈ cum_area."""
    from shapely.geometry import box as _sb
    minx, maxx = geom.bounds[0], geom.bounds[2]
    lo, hi = y_lo, y_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        a = geom.intersection(_sb(minx, y_lo, maxx, mid)).area
        if abs(a - cum_area) / (abs(cum_area) + 1e-10) < 1e-4:
            return mid
        lo, hi = (mid, hi) if a < cum_area else (lo, mid)
    return (lo + hi) / 2


# ── Rendering mappe ────────────────────────────────────────────────────────────
def _plot_choropleth(fig, ax, df_filt, geo_col, val_col, gdf, palette, title):
    """Mappa coropletica: regioni colorate per valore numerico."""
    _name_up = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    agg = df_filt.groupby(geo_col)[val_col].sum().reset_index()
    agg.columns = ["GEO_NAME", "value"]
    agg["value"] = pd.to_numeric(agg["value"], errors="coerce")
    agg["GEO_NAME"] = _normalize_geo(agg["GEO_NAME"], _name_up)

    merged = gdf.merge(agg, on="GEO_NAME", how="left")

    cmap_obj  = _get_cmap(palette, fallback="Blues")
    vmin = agg["value"].min()
    vmax = agg["value"].max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    merged.plot(
        column="value", ax=ax, cmap=cmap_obj,
        missing_kwds={"color": "#e4e4e4", "label": "Nessun dato"},
        legend=False, edgecolor="#aaaaaa", linewidth=0.5,
        vmin=vmin, vmax=vmax,
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = mcm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.88, top=0.97, bottom=0.02)


def _plot_bubble_map(fig, ax, df_filt, geo_col, val_col, grp_col, gdf, palette, title, show_legend):
    """Mappa bolla: cerchi proporzionali al valore ai centroidi."""
    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    centroids = gdf.set_index("GEO_NAME").geometry.centroid
    _name_up  = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    agg_cols  = [geo_col] + ([grp_col] if grp_col else [])
    if val_col:
        agg = df_filt.groupby(agg_cols)[val_col].sum().reset_index()
    else:
        agg = df_filt.groupby(agg_cols).size().reset_index(name="_n")
        val_col = "_n"

    agg["_GEO"] = _normalize_geo(agg[geo_col], _name_up)
    max_val     = pd.to_numeric(agg[val_col], errors="coerce").max() or 1

    if grp_col and grp_col in agg.columns:
        groups  = sorted(agg[grp_col].dropna().unique(), key=_xkey)
        cc      = _cat_colors(groups, palette)
        for grp in groups:
            sub = agg[agg[grp_col] == grp]
            for _, row in sub.iterrows():
                geo = row["_GEO"]
                if geo not in centroids.index:
                    continue
                cx, cy = centroids[geo].x, centroids[geo].y
                v = pd.to_numeric(row[val_col], errors="coerce")
                if pd.isna(v):
                    continue
                size = max(10, (v / max_val) * 1200)
                ax.scatter(cx, cy, s=size, color=cc[grp], alpha=0.65,
                           edgecolors="#333", linewidths=0.4, label=str(grp), zorder=4)
        if show_legend:
            handles = [plt.scatter([], [], s=60, color=cc[g], label=str(g), alpha=0.8)
                       for g in groups]
            ax.legend(handles=handles, fontsize=7, framealpha=0.9)
    else:
        base_color = _get_cmap(palette, fallback="tab10")(0)
        for _, row in agg.iterrows():
            geo = row["_GEO"]
            if geo not in centroids.index:
                continue
            cx, cy = centroids[geo].x, centroids[geo].y
            v = pd.to_numeric(row[val_col], errors="coerce")
            if pd.isna(v):
                continue
            size = max(10, (v / max_val) * 1200)
            ax.scatter(cx, cy, s=size, color=base_color, alpha=0.65,
                       edgecolors="#333", linewidths=0.4, zorder=4)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


def _plot_band_map(fig, ax, df_filt, geo_col, grp_col, val_col, gdf, palette, title, show_legend,
                   orientation="Verticale", equal_area=False):
    """
    Mappa bande: ogni area geografica è divisa in bande per categoria.
    orientation : "Verticale" (bande sinistra→destra) o "Orizzontale" (bande basso→alto)
    equal_area  : False = larghezza/altezza proporzionale al valore (default)
                  True  = area della banda proporzionale al valore
    """
    if not grp_col:
        raise ValueError("Mappa bande richiede 'Raggruppa/Colore'.")

    try:
        from shapely.geometry import box as shapely_box
    except ImportError:
        raise ImportError("shapely non installato: pip install shapely")

    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    _name_up = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    cats = sorted(df_filt[grp_col].dropna().unique(), key=_xkey)
    cc   = _cat_colors(cats, palette)

    df2 = df_filt.copy()
    df2["_GEO"] = _normalize_geo(df2[geo_col], _name_up)

    for _, prow in gdf.iterrows():
        gname = str(prow["GEO_NAME"])
        geom  = prow.geometry
        if geom is None or geom.is_empty:
            continue
        sub = df2[df2["_GEO"] == gname]
        if sub.empty:
            continue

        vals = {}
        for cat in cats:
            sub_c = sub[sub[grp_col] == cat]
            vals[cat] = (pd.to_numeric(sub_c[val_col], errors="coerce").sum()
                         if val_col else len(sub_c))
        total = sum(vals.values())
        if total == 0:
            continue

        minx, miny, maxx, maxy = geom.bounds

        if orientation == "Orizzontale":
            # Bande basso → alto (cut by Y)
            if equal_area:
                total_area = geom.area
                cum = 0.0
                y_prev = miny
                for i, cat in enumerate(cats):
                    if i == len(cats) - 1:
                        clip = shapely_box(minx, y_prev, maxx, maxy)
                    else:
                        cum += vals[cat]
                        y_next = _band_split_h(geom, total_area * (cum / total), miny, maxy)
                        clip   = shapely_box(minx, y_prev, maxx, y_next)
                        y_prev = y_next
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
            else:
                height = maxy - miny
                y_cur  = miny
                for cat in cats:
                    sh = height * (vals[cat] / total)
                    if sh < 1:
                        continue
                    clip = shapely_box(minx, y_cur, maxx, y_cur + sh)
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
                    y_cur += sh
        else:
            # Bande sinistra → destra (cut by X) — default
            if equal_area:
                total_area = geom.area
                cum = 0.0
                x_prev = minx
                for i, cat in enumerate(cats):
                    if i == len(cats) - 1:
                        clip = shapely_box(x_prev, miny, maxx, maxy)
                    else:
                        cum += vals[cat]
                        x_next = _band_split_v(geom, total_area * (cum / total), minx, maxx)
                        clip   = shapely_box(x_prev, miny, x_next, maxy)
                        x_prev = x_next
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
            else:
                width = maxx - minx
                x_cur = minx
                for cat in cats:
                    sw = width * (vals[cat] / total)
                    if sw < 1:
                        continue
                    clip = shapely_box(x_cur, miny, x_cur + sw, maxy)
                    frag = geom.intersection(clip)
                    if not frag.is_empty:
                        gpd.GeoSeries([frag], crs=gdf.crs).plot(
                            ax=ax, color=cc[cat], linewidth=0, zorder=4)
                    x_cur += sw

        gpd.GeoSeries([geom], crs=gdf.crs).plot(
            ax=ax, color="none", edgecolor="#555", linewidth=0.8, zorder=5)

    if show_legend:
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=cc[c], label=str(c)) for c in cats]
        ax.legend(handles=patches, fontsize=7, framealpha=0.9,
                  loc="lower right", title=grp_col)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


def _plot_pin_map(fig, ax, df_filt, geo_col, val_col, grp_col, gdf, palette, title, show_legend):
    """Mappa pin: marker ai centroidi, colorati per valore/gruppo."""
    gdf.plot(color="#e4e4e4", edgecolor="#aaaaaa", linewidth=0.5, ax=ax)
    centroids = gdf.set_index("GEO_NAME").geometry.centroid
    _name_up  = {n.upper().strip(): n for n in gdf["GEO_NAME"]}

    df2 = df_filt.copy()
    df2["_GEO"] = _normalize_geo(df2[geo_col], _name_up)

    if grp_col and grp_col in df2.columns:
        groups = sorted(df2[grp_col].dropna().unique(), key=_xkey)
        cc     = _cat_colors(groups, palette)
        for grp in groups:
            sub = df2[df2[grp_col] == grp]
            xs, ys = [], []
            for _, row in sub.iterrows():
                geo = row["_GEO"]
                if geo in centroids.index:
                    xs.append(centroids[geo].x)
                    ys.append(centroids[geo].y)
            if xs:
                ax.scatter(xs, ys, s=60, color=cc[grp], alpha=0.85,
                           edgecolors="#333", linewidths=0.5,
                           label=str(grp), zorder=4, marker="o")
        if show_legend:
            handles = [plt.scatter([], [], s=60, color=cc[g], label=str(g), alpha=0.85)
                       for g in groups]
            ax.legend(handles=handles, fontsize=7, framealpha=0.9)
    else:
        # Colora per valore se disponibile, altrimenti colore fisso
        base_color = _get_cmap(palette, fallback="tab10")(0.2)
        if val_col and val_col in df2.columns:
            agg = df2.groupby("_GEO")[val_col].sum().reset_index()
            vvals = pd.to_numeric(agg[val_col], errors="coerce")
            vmin, vmax = vvals.min(), vvals.max()
            cmap = _get_cmap(palette, fallback="YlOrRd")
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax != vmin else vmin + 1)
            for _, row in agg.iterrows():
                geo = row["_GEO"]
                if geo not in centroids.index:
                    continue
                v = pd.to_numeric(row[val_col], errors="coerce")
                if pd.isna(v):
                    continue
                c = cmap(norm(v))
                ax.scatter(centroids[geo].x, centroids[geo].y,
                           s=60, color=c, alpha=0.85,
                           edgecolors="#333", linewidths=0.5, zorder=4)
            sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        else:
            for gname, pt in centroids.items():
                ax.scatter(pt.x, pt.y, s=60, color=base_color, alpha=0.85,
                           edgecolors="#333", linewidths=0.5, zorder=4)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)


def _render_chart_on_ax(ax, df_sub, chart_key, x, y, g,
                         palette, show_nums, xtick_rot, data_lbl_sz, extra=None):
    """Renderizza grafico su ax; restituisce ax (può essere nuovo asse polare per radar)."""
    ex         = extra or {}
    horiz      = bool(ex.get("horiz",      False))
    stacked_3d = bool(ex.get("stacked_3d", False))
    show_line  = bool(ex.get("show_line",  False))
    line_col   = ex.get("line_col") or None
    podio_n    = max(1, min(10, int(ex.get("podio_n", 3))))
    is_freq    = (y == _FREQ_LABEL or not y)

    # ── BARRE (verticali / orizzontali) ──────────────────────────────────────
    if chart_key == "bar":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        if not is_freq and g:
            agg   = _agg(df_sub, x, y, g)
            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
            cols  = _colors(len(pivot.columns), palette)
            if horiz:
                pivot.plot.barh(ax=ax, color=cols)
            else:
                pivot.plot.bar(ax=ax, color=cols)
                _smart_labels(ax, pivot.index, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)
        elif not is_freq:
            agg_df, val_col = _agg_by_x(df_sub, x, y)
            vals = pd.to_numeric(agg_df[val_col], errors="coerce").fillna(0)
            cols = _colors(len(agg_df), palette)
            if horiz:
                ax.barh(range(len(agg_df)), vals, color=cols)
                ax.set_yticks(range(len(agg_df)))
                ax.set_yticklabels(agg_df[x].astype(str), fontsize=7)
            else:
                ax.bar(range(len(agg_df)), vals, color=cols)
                ax.set_xticks(range(len(agg_df)))
                _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
            if show_nums and ax.containers:
                ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=data_lbl_sz, padding=2)
        else:
            # Frequenza: con o senza g
            if g:
                frq = df_sub.groupby([x, g], sort=False).size().reset_index(name="Frequenza")
                pivot = frq.pivot_table(index=x, columns=g, values="Frequenza", fill_value=0)
                cols  = _colors(len(pivot.columns), palette)
                if horiz:
                    pivot.plot.barh(ax=ax, color=cols)
                else:
                    pivot.plot.bar(ax=ax, color=cols)
                    _smart_labels(ax, pivot.index, xtick_rot)
                if show_nums:
                    for cnt in ax.containers:
                        ax.bar_label(cnt, fmt="%d", fontsize=data_lbl_sz, padding=2)
            else:
                agg_df, _ = _agg_by_x(df_sub, x, _FREQ_LABEL)
                vals = pd.to_numeric(agg_df["Frequenza"], errors="coerce").fillna(0)
                cols = _colors(len(agg_df), palette)
                if horiz:
                    ax.barh(range(len(agg_df)), vals, color=cols)
                    ax.set_yticks(range(len(agg_df)))
                    ax.set_yticklabels(agg_df[x].astype(str), fontsize=7)
                else:
                    ax.bar(range(len(agg_df)), vals, color=cols)
                    ax.set_xticks(range(len(agg_df)))
                    _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
                if show_nums and ax.containers:
                    ax.bar_label(ax.containers[0], fmt="%d", fontsize=data_lbl_sz, padding=2)

    # ── BARRE IMPILATE (normale + 3D mattoncini) ──────────────────────────────
    elif chart_key == "bar_stacked":
        if not x:
            raise ValueError("Barre impilate richiedono almeno Asse X.")
        if stacked_3d:
            import matplotlib.patches as mpatches
            if not g:
                raise ValueError("La variante 3D (Pila mattoncini) richiede anche Raggruppa.")
            if not y:
                raise ValueError("La variante 3D richiede anche Asse Y.")
            cats   = sorted(df_sub[x].dropna().unique(), key=_xkey)
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            agg_d  = _agg(df_sub, x, y, g)
            _piv   = agg_d.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
            pivot  = _piv.reindex(
                index=cats,
                columns=[gr for gr in groups if gr in _piv.columns],
                fill_value=0)
            dx_, dy_, bar_w = 0.22, 0.11, 0.6
            ax.axis("off")
            max_val = pivot.sum(axis=1).max() or 1
            n_cats  = len(cats)
            y_scale = (n_cats * 1.2) / max_val

            def _shade(col_hex, factor):
                r_, g_, b_, a_ = mcolors.to_rgba(col_hex)
                return (min(1, r_ * factor), min(1, g_ * factor), min(1, b_ * factor), a_)

            for ci, cat in enumerate(cats):
                x0   = ci * 1.2
                base = 0.0
                for grp in groups:
                    if grp not in pivot.columns:
                        continue
                    val = float(pivot.loc[cat, grp]) if cat in pivot.index else 0.0
                    if val <= 0:
                        continue
                    h  = val * y_scale
                    b_ = base * y_scale
                    c  = cc[grp]
                    ax.add_patch(mpatches.Polygon(
                        [[x0, b_], [x0+bar_w, b_], [x0+bar_w, b_+h], [x0, b_+h]],
                        closed=True, facecolor=c, edgecolor="white", linewidth=0.6, zorder=3))
                    ax.add_patch(mpatches.Polygon(
                        [[x0, b_+h], [x0+bar_w, b_+h],
                         [x0+bar_w+dx_, b_+h+dy_], [x0+dx_, b_+h+dy_]],
                        closed=True, facecolor=_shade(c, 1.35), edgecolor="white",
                        linewidth=0.6, zorder=3))
                    ax.add_patch(mpatches.Polygon(
                        [[x0+bar_w, b_], [x0+bar_w+dx_, b_+dy_],
                         [x0+bar_w+dx_, b_+h+dy_], [x0+bar_w, b_+h]],
                        closed=True, facecolor=_shade(c, 0.65), edgecolor="white",
                        linewidth=0.6, zorder=3))
                    base += val
                ax.text(x0 + bar_w / 2, -0.15, str(cat),
                        ha="center", va="top", fontsize=8, color="#333")
            handles = [mpatches.Patch(facecolor=cc[gr], label=str(gr))
                       for gr in groups if gr in cc]
            if handles:
                ax.legend(handles=handles, fontsize=8, loc="upper right")
            total_w = (n_cats - 1) * 1.2 + bar_w + dx_ + 0.3
            ax.set_xlim(-0.3, total_w)
            ax.set_ylim(-0.4, max_val * y_scale * 1.25 + dy_ + 0.2)
        else:
            if not (y and g):
                raise ValueError("Barre impilate richiedono Asse X, Asse Y e Raggruppa.")
            agg   = _agg(df_sub, x, y, g)
            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
            pivot.plot.bar(ax=ax, stacked=True, color=_colors(len(pivot.columns), palette))
            _smart_labels(ax, pivot.index, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)

    # ── LINEE ─────────────────────────────────────────────────────────────────
    elif chart_key == "line":
        if not x:
            raise ValueError("Seleziona almeno Asse X.")
        x_all = sorted(df_sub[x].dropna().unique(), key=_xkey)
        x_pos = {str(v): i for i, v in enumerate(x_all)}
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            colors = _colors(len(groups), palette)
            for grp, col_ in zip(groups, colors):
                sub            = df_sub[df_sub[g] == grp]
                agg_sub, vcol  = _agg_by_x(sub, x, y)
                xvals = [x_pos.get(str(v)) for v in agg_sub[x]]
                yvals = pd.to_numeric(agg_sub[vcol], errors="coerce").tolist()
                pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                         if xv is not None and not pd.isna(yv)]
                if pairs:
                    xs_, ys_ = zip(*pairs)
                    ax.plot(list(xs_), list(ys_), marker="o", markersize=4,
                            label=str(grp), color=col_)
        else:
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            xvals = [x_pos.get(str(v)) for v in agg_df[x]]
            yvals = pd.to_numeric(agg_df[vcol], errors="coerce").tolist()
            pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                     if xv is not None and not pd.isna(yv)]
            if pairs:
                xs_, ys_ = zip(*pairs)
                ax.plot(list(xs_), list(ys_), marker="o", markersize=4,
                        color=_colors(3, palette)[1])
        ax.set_xticks(range(len(x_all)))
        _smart_labels(ax, [str(v) for v in x_all], xtick_rot)

    # ── DISPERSIONE ───────────────────────────────────────────────────────────
    elif chart_key == "scatter":
        if not (x and y):
            raise ValueError("Seleziona Asse X e Asse Y numerici.")
        xv = pd.to_numeric(df_sub[x], errors="coerce")
        yv = pd.to_numeric(df_sub[y], errors="coerce")
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            for grp in groups:
                mask = df_sub[g] == grp
                col_ = cc[grp]
                ax.scatter(xv[mask], yv[mask], s=40, alpha=0.7,
                           color=col_, edgecolors="white", linewidths=0.4,
                           label=str(grp))
                if show_line:
                    tmp = pd.DataFrame({"xv": xv[mask], "yv": yv[mask]}).dropna().sort_values("xv")
                    if len(tmp) >= 2:
                        ax.plot(tmp["xv"].values, tmp["yv"].values,
                                color=col_, alpha=0.5, linewidth=1)
        else:
            col_ = _colors(3, palette)[1]
            ax.scatter(xv, yv, s=40, alpha=0.7, color=col_,
                       edgecolors="white", linewidths=0.4)
            if show_line:
                tmp = pd.DataFrame({"xv": xv, "yv": yv}).dropna().sort_values("xv")
                if len(tmp) >= 2:
                    ax.plot(tmp["xv"].values, tmp["yv"].values,
                            color=col_, alpha=0.5, linewidth=1)

    # ── ISTOGRAMMA ────────────────────────────────────────────────────────────
    elif chart_key == "histogram":
        col_ = y or x
        if not col_:
            raise ValueError("Seleziona una colonna numerica.")
        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            colors = _colors(len(groups), palette)
            for grp, clr in zip(groups, colors):
                vals = pd.to_numeric(
                    df_sub.loc[df_sub[g] == grp, col_], errors="coerce"
                ).dropna()
                ax.hist(vals, bins=20, alpha=0.55, label=str(grp), color=clr)
        else:
            vals = pd.to_numeric(df_sub[col_], errors="coerce").dropna()
            ax.hist(vals, bins=20, color=_colors(3, palette)[1])

    # ── TORTA ─────────────────────────────────────────────────────────────────
    elif chart_key == "pie":
        if not x:
            raise ValueError("Seleziona Asse X (categorie).")
        agg_df, vcol = _agg_by_x(df_sub, x, y)
        agg_df = agg_df.sort_values(vcol, ascending=False)
        top    = agg_df.head(12)
        vals   = pd.to_numeric(top[vcol], errors="coerce").fillna(0)
        _, _, autotexts = ax.pie(
            vals.values,
            labels=[str(l) for l in top[x]],
            autopct="%1.1f%%", startangle=90,
            colors=_colors(len(top), palette),
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.axis("equal")

    # ── HEATMAP (3 input: x=colonne, g=righe, y=valori/colore) ───────────────
    elif chart_key == "heatmap":
        if not (x and g):
            raise ValueError("Heatmap richiede Asse X1 (colonne) e Asse X2 (righe).")
        if is_freq:
            piv = (df_sub.groupby([g, x], sort=False).size()
                   .unstack(x).fillna(0))
            cbar_lbl = "Frequenza"
        else:
            piv = df_sub.pivot_table(index=g, columns=x, values=y,
                                     aggfunc="mean", fill_value=0)
            cbar_lbl = str(y)
        if piv.empty:
            raise ValueError("Heatmap vuota: nessun dato con le selezioni correnti.")
        cmap   = _get_cmap(palette, fallback="YlOrRd")
        vmax   = float(piv.values.max()) or 1.0
        im     = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        cols_x = list(range(len(piv.columns)))
        rows_y = list(range(len(piv.index)))
        ax.set_xticks(cols_x)
        ax.set_xticklabels([str(c) for c in piv.columns],
                            rotation=45, ha="right", fontsize=7)
        ax.set_yticks(rows_y)
        ax.set_yticklabels([str(r) for r in piv.index], fontsize=7)
        for i in rows_y:
            for j in cols_x:
                v      = piv.iloc[i, j]
                txtcol = "white" if (v / vmax) > 0.55 else "#111"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=6, color=txtcol)
        cb = ax.get_figure().colorbar(im, ax=ax, shrink=0.7)
        cb.set_label(cbar_lbl, fontsize=8)

    # ── BOXPLOT ───────────────────────────────────────────────────────────────
    elif chart_key == "boxplot":
        if not y:
            raise ValueError("Seleziona Asse Y (colonna numerica).")
        mp = dict(color="black", linewidth=1.5)
        if x:
            cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
            data = [pd.to_numeric(
                df_sub.loc[df_sub[x] == c, y], errors="coerce"
            ).dropna() for c in cats]
            bp = ax.boxplot(data, patch_artist=True, medianprops=mp)
            for patch, col_ in zip(bp["boxes"], _colors(len(cats), palette)):
                patch.set_facecolor(col_)
            ax.set_xticks(range(1, len(cats) + 1))
            _smart_labels(ax, [str(c) for c in cats], xtick_rot)
        else:
            vals = pd.to_numeric(df_sub[y], errors="coerce").dropna()
            bp   = ax.boxplot(vals, patch_artist=True, medianprops=mp)
            bp["boxes"][0].set_facecolor(_colors(3, palette)[1])

    # ── RADAR ─────────────────────────────────────────────────────────────────
    elif chart_key == "radar":
        if not (x and y):
            raise ValueError("Radar richiede Asse X (categorie) e Asse Y (valori).")
        cats = sorted(df_sub[x].dropna().unique(), key=_xkey)
        N    = len(cats)
        if N < 3:
            raise ValueError("Il radar richiede almeno 3 categorie nell'Asse X.")
        angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles_c = angles + angles[:1]
        # rimpiazza asse normale con asse polare (restituito alla fine)
        fig_  = ax.get_figure()
        pos   = ax.get_position()
        ax.remove()
        ax = fig_.add_axes([pos.x0, pos.y0, pos.width, pos.height], polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        def _mean_per_cat(sub):
            return [float(pd.to_numeric(
                sub.loc[sub[x] == cat, y], errors="coerce"
            ).mean() or 0) for cat in cats]

        if g:
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            for grp in groups:
                vals_c = _mean_per_cat(df_sub[df_sub[g] == grp]) + [0]
                vals_c[-1] = vals_c[0]
                c = cc[grp]
                ax.plot(angles_c, vals_c, color=c, linewidth=2, label=str(grp))
                ax.fill(angles_c, vals_c, alpha=0.15, color=c)
        else:
            vals_c = _mean_per_cat(df_sub) + [0]
            vals_c[-1] = vals_c[0]
            c = _colors(3, palette)[1]
            ax.plot(angles_c, vals_c, color=c, linewidth=2)
            ax.fill(angles_c, vals_c, alpha=0.2, color=c)
        ax.set_xticks(angles)
        ax.set_xticklabels([str(c) for c in cats], fontsize=8)
        ax.grid(True, alpha=0.35)

    # ── IMBUTO (FUNNEL) ───────────────────────────────────────────────────────
    elif chart_key == "funnel":
        if not x:
            raise ValueError("Seleziona Asse X (etichette) per l'imbuto.")
        import matplotlib.patches as mpatches
        cx, max_h, min_h = 0.5, 0.42, 0.07

        def _conn_f(dst, src, col_):
            ax.annotate("", xy=dst, xytext=src,
                        arrowprops=dict(arrowstyle="-", color=col_, lw=0.85, alpha=0.65), zorder=3)

        if g and not is_freq:
            # ── Funnel con Raggruppa: ogni trapezio è diviso in segmenti per gruppo ──
            cats   = sorted(df_sub[x].dropna().unique(), key=_xkey)
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            totals = {cat: float(pd.to_numeric(df_sub.loc[df_sub[x]==cat, y if y else "_"],
                                               errors="coerce").sum() if y else len(df_sub[df_sub[x]==cat]))
                      for cat in cats}
            # somma reale: se y è una colonna, somma; altrimenti conta le righe
            if y and y != _FREQ_LABEL:
                totals = {cat: float(pd.to_numeric(df_sub.loc[df_sub[x]==cat, y],
                                                   errors="coerce").sum()) for cat in cats}
            else:
                totals = {cat: float((df_sub[x] == cat).sum()) for cat in cats}
            cats   = sorted(cats, key=lambda c: totals.get(c, 0), reverse=True)
            N      = len(cats)
            if N < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            if y and y != _FREQ_LABEL:
                piv = (df_sub.groupby([x, g])[y].sum()).unstack(g).reindex(cats).fillna(0)
            else:
                piv = (df_sub.groupby([x, g]).size()).unstack(g).reindex(cats).fillna(0)
            piv    = piv.reindex(columns=[gr for gr in groups if gr in piv.columns])
            v_max  = max(totals.values()) or 1
            halfs  = [min_h + (totals.get(c, 0) / v_max) * (max_h - min_h) for c in cats]
            cc     = _cat_colors(groups, palette)
            ax.axis("off"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.3, N + 0.3)
            for i, cat in enumerate(cats):
                y_top = N - i;  y_bot = N - i - 1;  y_mid = (y_top + y_bot) / 2
                h_top = halfs[i];  h_bot = halfs[i+1] if i+1 < N else min_h
                row_tot = float(piv.loc[cat].sum()) if cat in piv.index else 1.0
                if row_tot <= 0:
                    row_tot = 1.0
                off_top = cx - h_top;  off_bot = cx - h_bot
                full_top = 2 * h_top;  full_bot = 2 * h_bot
                for grp in groups:
                    val = float(piv.loc[cat, grp]) if (cat in piv.index and grp in piv.columns) else 0.0
                    if val <= 0:
                        continue
                    frac   = val / row_tot
                    sw_top = frac * full_top;  sw_bot = frac * full_bot
                    col_   = cc[grp]
                    ax.add_patch(mpatches.Polygon(
                        [[off_top, y_top], [off_top+sw_top, y_top],
                         [off_bot+sw_bot, y_bot], [off_bot, y_bot]],
                        closed=True, facecolor=col_, edgecolor="white", linewidth=0.8, zorder=2))
                    off_top += sw_top;  off_bot += sw_bot
                ax.plot([cx-h_top, cx+h_top], [y_top, y_top],
                        color="white", linewidth=1.2, zorder=4)
                pct = totals.get(cat, 0) / v_max * 100
                _conn_f((-0.04, y_mid), (cx-h_top, y_mid), "#999")
                ax.text(-0.08, y_mid, str(cat),
                        ha="right", va="center", fontsize=9, fontweight="bold",
                        color="#1a1a1a", zorder=5)
                _conn_f((1.04, y_mid), (cx+h_top, y_mid), "#999")
                ax.text(1.08, y_mid, f"{int(round(totals.get(cat,0))):,}  ({pct:.1f} %)",
                        ha="left", va="center", fontsize=9, color="#1a1a1a", zorder=5)
            handles = [mpatches.Patch(facecolor=cc[gr], label=str(gr))
                       for gr in groups if gr in cc]
            if handles:
                ax.legend(handles=handles, fontsize=8, loc="lower right")
        else:
            # ── Funnel semplice (un colore per livello) ────────────────────────
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            agg_df  = agg_df.sort_values(vcol, ascending=False).reset_index(drop=True)
            labels  = list(agg_df[x].astype(str))
            values  = list(pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0))
            N = len(labels)
            if N < 2:
                raise ValueError("L'imbuto richiede almeno 2 categorie.")
            colors = _colors(N, palette)
            v_max  = max(values) or 1
            halfs  = [min_h + (v / v_max) * (max_h - min_h) for v in values]
            ax.axis("off"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.3, N + 0.3)
            for i in range(N):
                col_   = colors[i]
                y_top  = N - i;  y_bot = N - i - 1;  y_mid = (y_top + y_bot) / 2
                h_top  = halfs[i];  h_bot = halfs[i + 1] if i + 1 < N else min_h
                r0, g0, b0, _ = mcolors.to_rgba(col_)
                hi = (min(1, r0*1.4), min(1, g0*1.4), min(1, b0*1.4), 0.7)
                ax.add_patch(mpatches.Polygon(
                    [[cx-h_top, y_top], [cx+h_top, y_top],
                     [cx+h_bot, y_bot], [cx-h_bot, y_bot]],
                    closed=True, facecolor=col_, edgecolor="white", linewidth=0.8, zorder=2))
                ax.plot([cx-h_top, cx+h_top], [y_top, y_top],
                        color=hi, linewidth=2.5, zorder=4, solid_capstyle="round")
                _conn_f((-0.04, y_mid), (cx-h_top, y_mid), col_)
                ax.text(-0.08, y_mid, str(labels[i]),
                        ha="right", va="center", fontsize=9, fontweight="bold",
                        color="#1a1a1a", zorder=5)
                pct = values[i] / values[0] * 100 if values[0] else 0
                _conn_f((1.04, y_mid), (cx+h_top, y_mid), col_)
                ax.text(1.08, y_mid, f"{int(round(values[i])):,}  ({pct:.1f} %)",
                        ha="left", va="center", fontsize=9, color="#1a1a1a", zorder=5)

    # ── PODIO ─────────────────────────────────────────────────────────────────
    elif chart_key == "podio":
        if not x:
            raise ValueError("Seleziona Asse X (nomi) per il podio.")
        from matplotlib.patches import Rectangle
        if y and y in df_sub.columns and y != _FREQ_LABEL:
            nums = pd.to_numeric(df_sub[y], errors="coerce")
            df2  = df_sub.assign(_score=nums).dropna(subset=["_score"])
            df2  = df2.sort_values("_score", ascending=False).reset_index(drop=True)
            score_col = y
        else:
            df2 = df_sub.copy().reset_index(drop=True)
            df2["_score"] = (len(df2) - df2.index).astype(float)
            score_col = None
        n = min(len(df2), podio_n)
        if n == 0:
            raise ValueError("Nessun dato per il podio.")
        df2 = df2.iloc[:n].copy()

        _MEDAL   = [("#D4AF37","#9A7600"), ("#B0B0BC","#6A6A80"), ("#CD7F32","#8B4513")]
        _extra_c = _colors(max(1, n), palette)
        ORDINALS = ("1°","2°","3°","4°","5°","6°","7°","8°","9°","10°")
        _STEP_H  = {0: 1.00, 1: 0.70, 2: 0.50}

        def _gc(ri):
            return _MEDAL[ri] if ri < 3 else (_extra_c[ri % len(_extra_c)], "#555555")
        def _sh(ri):
            return _STEP_H.get(ri, max(0.25, 0.50 - (ri - 2) * 0.06))

        _OFF  = (0, 1, -1, -2, 2, -3, 3, -4, 4, -5)
        ctr   = -min(_OFF[:n])
        col4r = [ctr + _OFF[i] for i in range(n)]
        col_w = 1.0;  sw = 0.86;  card_h = 0.58;  cgap = 0.05
        strip_h = 0.085;  base_h = 0.05;  white_h = card_h - strip_h
        _fs_n = max(4.5, 7.5 - n * 0.22);  _fs_d = max(3.5, 5.8 - n * 0.18)
        _fs_o = max(5.5, 7.5 - n * 0.2)

        ax.set_facecolor("#F0F0F0")
        for ri in range(n):
            ci = col4r[ri];  cx_p = ci*col_w + col_w/2;  x0_p = cx_p - sw/2
            sh = _sh(ri);  fc, ec = _gc(ri)
            ax.add_patch(Rectangle((x0_p, 0), sw, sh, facecolor=fc,
                                    edgecolor=ec, linewidth=1.5, zorder=3))
            ax.add_patch(Rectangle((x0_p, sh-0.022), sw, 0.022,
                                    facecolor="white", alpha=0.22, linewidth=0, zorder=4))
            ax.text(cx_p, sh*0.40, str(ri+1), ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white", alpha=0.28, zorder=4)
            cy = sh + cgap
            ax.add_patch(Rectangle((x0_p, cy), sw, card_h, facecolor="white",
                                    edgecolor=ec, linewidth=2.0, zorder=5))
            ax.add_patch(Rectangle((x0_p, cy+white_h), sw, strip_h,
                                    facecolor=fc, linewidth=0, zorder=6))
            ax.text(cx_p, cy+white_h+strip_h/2, ORDINALS[ri],
                    ha="center", va="center", fontsize=_fs_o,
                    fontweight="bold", color="white", zorder=7)
            name = str(df2.iloc[ri][x]) if x in df2.columns else f"#{ri+1}"
            if len(name) > 30:
                name = name[:29] + "…"
            nw = "\n".join(textwrap.wrap(name, 12)[:2]) or name
            details = []
            if score_col and score_col in df2.columns:
                sv = df2.iloc[ri]["_score"]
                try:
                    sv_s = f"{sv:,.0f}" if float(sv)==int(float(sv)) else f"{sv:.4g}"
                except Exception:
                    sv_s = str(sv)
                details.append(f"{score_col[:12]}: {sv_s}")
            if g and g in df2.columns:
                details.append(str(df2.iloc[ri][g])[:18])
            ny = cy + white_h * (0.68 if details else 0.50)
            ax.text(cx_p, ny, nw, ha="center", va="center",
                    fontsize=_fs_n, fontweight="bold", color="#111111",
                    zorder=7, multialignment="center", clip_on=True)
            if details:
                ax.text(cx_p, cy+white_h*0.18, "\n".join(details),
                        ha="center", va="center", fontsize=_fs_d,
                        color="#555555", style="italic", zorder=7,
                        multialignment="center", clip_on=True)
            if ri == 0:
                ax.text(cx_p, cy+card_h+0.025, "★",
                        ha="center", va="bottom", fontsize=16,
                        color="#D4AF37", fontweight="bold", zorder=7)
        ax.add_patch(Rectangle((-0.07, -base_h), n*col_w+0.14, base_h,
                                facecolor="#2A2A2A", edgecolor="#111111",
                                linewidth=1, zorder=2))
        ax.set_xlim(-0.2, n*col_w+0.2)
        ax.set_ylim(-base_h-0.03, 1.0+card_h+cgap+0.22)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

    # ── VIOLINO ───────────────────────────────────────────────────────────────
    elif chart_key == "violin":
        if not y:
            raise ValueError("Seleziona Asse Y (valori numerici) per il violino.")
        if x:
            cats      = sorted(df_sub[x].dropna().unique(), key=_xkey)
            cc        = _cat_colors(cats, palette)
            data_sets = [pd.to_numeric(df_sub.loc[df_sub[x]==c, y],
                                       errors="coerce").dropna().values for c in cats]
            labels    = [str(c) for c in cats]
            clrs      = [cc[c] for c in cats]
        elif g:
            groups    = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc        = _cat_colors(groups, palette)
            data_sets = [pd.to_numeric(df_sub.loc[df_sub[g]==gr, y],
                                       errors="coerce").dropna().values for gr in groups]
            labels    = [str(gr) for gr in groups]
            clrs      = [cc[gr] for gr in groups]
        else:
            data_sets = [pd.to_numeric(df_sub[y], errors="coerce").dropna().values]
            labels    = [str(y)];  clrs = [_colors(3, palette)[1]]
        valid = [(d, l, c) for d, l, c in zip(data_sets, labels, clrs) if len(d) >= 2]
        if not valid:
            raise ValueError("Dati insufficienti per il violino (min 2 valori per categoria).")
        data_arr, labels, clrs = zip(*valid)
        positions = list(range(len(data_arr)))
        vp = ax.violinplot(list(data_arr), positions=positions,
                           showmedians=True, showextrema=True, widths=0.7)
        for body, c in zip(vp["bodies"], clrs):
            body.set_facecolor(c);  body.set_edgecolor(c);  body.set_alpha(0.65)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color("#444");  vp[part].set_linewidth(1.2)
        ax.set_xticks(positions)
        _smart_labels(ax, list(labels), xtick_rot)

    # ── BARRE + LINEE (COMBO) ─────────────────────────────────────────────────
    elif chart_key == "combo":
        if not x:
            raise ValueError("Combo richiede Asse X.")

        if g and y and not is_freq:
            # ── Barre raggruppate per g ────────────────────────────────────────
            cats   = sorted(df_sub[x].dropna().unique(), key=_xkey)
            groups = sorted(df_sub[g].dropna().unique(), key=_xkey)
            cc     = _cat_colors(groups, palette)
            agg_d  = _agg(df_sub, x, y, g)
            pivot  = agg_d.pivot_table(index=x, columns=g, values=y,
                                       aggfunc="sum").reindex(cats).fillna(0)
            bar_colors = [cc.get(gr, "#888") for gr in pivot.columns]
            pivot.plot.bar(ax=ax, color=bar_colors, alpha=0.82)
            _smart_labels(ax, pivot.index, xtick_rot)
            if show_nums:
                for cnt in ax.containers:
                    ax.bar_label(cnt, fmt="%.4g", fontsize=data_lbl_sz, padding=2)
            x_idx = {str(c): i for i, c in enumerate(cats)}
        else:
            # ── Barre semplici ─────────────────────────────────────────────────
            agg_df, vcol = _agg_by_x(df_sub, x, y)
            vals   = pd.to_numeric(agg_df[vcol], errors="coerce").fillna(0)
            colors = _colors(len(agg_df), palette)
            ax.bar(range(len(agg_df)), vals, color=colors, alpha=0.82)
            ax.set_xticks(range(len(agg_df)))
            _smart_labels(ax, agg_df[x].astype(str), xtick_rot)
            if show_nums and ax.containers:
                ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=data_lbl_sz, padding=2)
            x_idx = {str(v): i for i, v in enumerate(agg_df[x])}

        # ── Linea su asse Y2 separato ──────────────────────────────────────────
        if line_col and line_col in df_sub.columns:
            agg_l, lcol = _agg_by_x(df_sub, x, line_col)
            lvals = pd.to_numeric(agg_l[lcol], errors="coerce").fillna(0)
            lx    = [x_idx.get(str(v)) for v in agg_l[x]]
            pairs = [(xi, yv) for xi, yv in zip(lx, lvals) if xi is not None]
            if pairs:
                px, py = zip(*pairs)
                ax2 = ax.twinx()
                ax2.plot(list(px), list(py), color="#E63946", linewidth=2.2,
                         marker="o", markersize=5, zorder=5)
                ax2.set_ylabel(str(line_col), fontsize=9, color="#E63946")
                ax2.tick_params(axis="y", labelcolor="#E63946")
                # scala Y2 esplicita basata su min/max della linea
                lv_min, lv_max = float(lvals.min()), float(lvals.max())
                pad = (lv_max - lv_min) * 0.12 or abs(lv_min) * 0.1 or 1.0
                ax2.set_ylim(lv_min - pad, lv_max + pad)

    else:
        raise ValueError(f"Tipo grafico non supportato: {chart_key!r}")

    return ax


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGAZIONE  (radio orizzontale — permette st.rerun() per cambiare pagina)
# ══════════════════════════════════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# Valori di default per le opzioni di personalizzazione
_OPT_DEFAULTS = {
    # Testi
    "opt_title":        "",
    "opt_title_sz":     13,
    "opt_title_bold":   True,
    "opt_title_ital":   False,
    "opt_xlabel":       "",
    "opt_ylabel":       "",
    "opt_label_sz":     10,
    # Stile
    "opt_palette_nm":   list(PALETTES.keys())[0],
    "opt_show_leg":     True,
    "opt_show_grid":    True,
    "opt_show_nums":    False,
    "opt_data_lbl_sz":  7,
    "opt_minimal":      False,
    "opt_show_axes_y":  True,
    "opt_white_bg":     False,
    # Font tick e legenda
    "opt_xtick_sz":     8,
    "opt_ytick_sz":     8,
    "opt_legend_sz":    8,
    # Assi base
    "opt_xtick_rot":    -1,
    "opt_y_min":        "",
    "opt_y_max":        "",
    # Scala
    "opt_y_scale":      "Normale",
    "opt_x_scale":      "Normale",
    "opt_y_step":       "",
    "opt_x_step":       "",
    # Minor tick
    "opt_y_minor":      False,
    "opt_y_minor_ndiv": 5,
    "opt_x_minor":      False,
    "opt_x_minor_ndiv": 5,
    # Griglia avanzata Y (major + minor)
    "opt_gy_maj":       False,
    "opt_gy_maj_sty":   "-- tratteggiata",
    "opt_gy_maj_alp":   0.4,
    "opt_gy_maj_lw":    0.8,
    "opt_gy_min":       False,
    "opt_gy_min_sty":   ":  punteggiata",
    "opt_gy_min_alp":   0.25,
    "opt_gy_min_lw":    0.5,
    "opt_gy_min_ndiv":  5,
    # Griglia avanzata X (major + minor)
    "opt_gx_maj":       False,
    "opt_gx_maj_sty":   "-- tratteggiata",
    "opt_gx_maj_alp":   0.4,
    "opt_gx_maj_lw":    0.8,
    "opt_gx_min":       False,
    "opt_gx_min_sty":   ":  punteggiata",
    "opt_gx_min_alp":   0.25,
    "opt_gx_min_lw":    0.5,
    "opt_gx_min_ndiv":  5,
    # Esportazione
    "opt_dpi":          200,
    "opt_fig_w":        11.0,
    "opt_fig_h":        6.0,
}
for _k, _v in _OPT_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_TABS = ["📁  Dati", "🗂  Variabili", "📊  Grafico", "⚙️  Personalizzazione"]
_sel = st.radio(
    "", _TABS, horizontal=True,
    index=st.session_state.active_tab,
    label_visibility="collapsed",
)
st.session_state.active_tab = _TABS.index(_sel)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 1 — DATI
# ══════════════════════════════════════════════════════════════════════════════
if _sel == _TABS[0]:

    st.subheader("Carica file dati")
    uploaded = st.file_uploader(
        "Trascina qui il file oppure clicca per cercarlo",
        type=["xlsx", "xls", "xlsm", "csv"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_new = pd.read_csv(uploaded)
                label  = uploaded.name
            else:
                xf     = pd.ExcelFile(uploaded)
                sheet  = st.selectbox("Foglio", xf.sheet_names, key="sheet_sel")
                df_new = pd.read_excel(uploaded, sheet_name=sheet)
                label  = f"{uploaded.name} [{sheet}]"

            st.success(f"✓  {label}  —  {len(df_new):,} righe × {len(df_new.columns)} colonne")
            st.session_state.df = df_new
            st.session_state.col_roles = {c: "mostra" for c in df_new.columns}
            st.session_state.fig = None

        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(200), use_container_width=True)
        st.divider()
        if st.button("🗂  Vai alle Variabili →", use_container_width=True):
            st.session_state.active_tab = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 2 — VARIABILI
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[1]:
    df = st.session_state.df
    if df is None:
        st.info("Carica prima un file nella sezione Dati.")
    else:
        st.subheader("Colonne visibili")
        st.caption("**Mostra** = colonna disponibile per assi e filtri  |  **Ignora** = esclusa")

        roles        = st.session_state.col_roles.copy()
        ROLE_OPTIONS = ["mostra", "ignora"]

        # ── Azioni rapide ──────────────────────────────────────────────────────
        qa, qb = st.columns(2)
        with qa:
            if st.button("✅  Mostra tutto", use_container_width=True):
                for col in df.columns:
                    roles[col] = "mostra"
                st.session_state.col_roles = roles
                st.rerun()
        with qb:
            if st.button("🚫  Ignora tutto", use_container_width=True):
                for col in df.columns:
                    roles[col] = "ignora"
                st.session_state.col_roles = roles
                st.rerun()

        st.divider()

        # ── Selettori per colonna (griglia 4 per riga) ────────────────────────
        N_COLS   = 4
        col_list = list(df.columns)
        for row_start in range(0, len(col_list), N_COLS):
            row_widgets = st.columns(N_COLS)
            for j, col_name in enumerate(col_list[row_start : row_start + N_COLS]):
                with row_widgets[j]:
                    cur  = roles.get(col_name, "mostra")
                    if cur not in ROLE_OPTIONS:
                        cur = "mostra"
                    icon = "🔢" if _is_num(df[col_name]) else "🔤"
                    new_role = st.selectbox(
                        f"{icon} {col_name}", ROLE_OPTIONS,
                        index=ROLE_OPTIONS.index(cur),
                        key=f"role_{col_name}",
                    )
                    roles[col_name] = new_role

        st.session_state.col_roles = roles

        show_cols = [c for c, r in roles.items() if r == "mostra"]
        if show_cols:
            st.divider()
            st.subheader("Anteprima dati")
            st.dataframe(df[show_cols].head(300), use_container_width=True)

        st.divider()
        if st.button("📊  Vai al Grafico →", use_container_width=True):
            st.session_state.active_tab = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 3 — GRAFICO
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[2]:
    df = st.session_state.df
    if df is None:
        st.info("Carica prima un file nella sezione Dati.")
    else:
        roles   = st.session_state.col_roles
        visible = [""] + [c for c, r in roles.items() if r == "mostra"]
        indip   = visible
        dipend  = visible
        all_col = visible

        left, right = st.columns([1, 2])

        with left:
            ss = st.session_state

            # ── Stato persistente Tab Grafico (pattern: value= senza key=) ────
            _g3_init = {
                "g3_chart":     None,
                "g3_scope":     None,
                "g3_geo_col":   "",
                "g3_x_col":     "",
                "g3_y_col":     "",
                "g3_grp_col":   "",
                "g3_n_filt":    0,
                "g3_n_sep":     0,
                "g3_filters":   [{"col": "", "type": "Valori (lista)", "val": []}
                                  for _ in range(8)],
                "g3_seps":      [{"col": "", "mode": "cumulativo", "fval": ""}
                                  for _ in range(6)],
                # opzioni specifiche per tipo grafico
                "g3_bar_horiz": False,
                "g3_bar_3d":    False,
                "g3_show_line": False,
                "g3_line_col":  "",
                "g3_podio_n":   3,
            }
            for _k, _v in _g3_init.items():
                if _k not in ss:
                    ss[_k] = _v

            # ── Tipo grafico ──────────────────────────────────────────────────
            st.subheader("Tipo di grafico")
            chart_labels = [t[0] for t in CHART_TYPES]
            chart_keys   = [t[1] for t in CHART_TYPES]
            _cl_idx      = chart_labels.index(ss.g3_chart) if ss.g3_chart in chart_labels else 0
            chart_label  = st.selectbox("", chart_labels, index=_cl_idx, label_visibility="collapsed")
            ss.g3_chart  = chart_label
            chart_key    = chart_keys[chart_labels.index(chart_label)]
            is_map       = chart_key in MAP_KEYS

            # ── Opzioni mappa ─────────────────────────────────────────────────
            if is_map:
                st.subheader("Impostazioni mappa")
                if not GEO_OK:
                    st.error("geopandas non installato — le mappe non sono disponibili.")
                _sc_opts    = [s[0] for s in MAP_SCOPES]
                _sc_idx     = _sc_opts.index(ss.g3_scope) if ss.g3_scope in _sc_opts else 0
                scope_label = st.selectbox("Ambito mappa", _sc_opts, index=_sc_idx)
                ss.g3_scope = scope_label
                scope_key   = MAP_SCOPE_KEYS[scope_label]
                _geo_opts   = [""] + list(df.columns)
                _geo_idx    = _geo_opts.index(ss.g3_geo_col) if ss.g3_geo_col in _geo_opts else 0
                geo_col     = st.selectbox(
                    "Colonna geo (nomi geografici nel tuo file)", _geo_opts, index=_geo_idx,
                    help="Seleziona la colonna che contiene i nomi delle aree geografiche",
                )
                ss.g3_geo_col = geo_col
                if chart_key == "band_map":
                    val_col_map     = st.selectbox("Valore (opzionale, per proporzionare bande)", dipend)
                    grp_col_map     = st.selectbox("Categoria colore (obbligatorio)", all_col)
                    band_orient     = st.radio(
                        "Direzione bande", ["Verticale (→)", "Orizzontale (↑)"], horizontal=True,
                    )
                    band_equal_area = st.checkbox(
                        "Area proporzionale al valore", value=False,
                        help=(
                            "Se attivo: l'area della banda nel poligono è proporzionale al valore "
                            "(ricerca binaria). Se disattivo: la larghezza/altezza del bounding box "
                            "è proporzionale al valore (più veloce, meno preciso per forme irregolari)."
                        ),
                    )
                elif chart_key in ("bubble_map", "pin_map"):
                    val_col_map = st.selectbox("Valore (dimensione/colore)", dipend)
                    grp_col_map = st.selectbox("Raggruppa/Colore (opzionale)", all_col)
                else:  # choropleth
                    val_col_map = st.selectbox("Valore numerico (Asse Y)", dipend)
                    grp_col_map = ""

            # ── Assi (solo grafici non-mappa) ─────────────────────────────────
            if not is_map:
                st.subheader("Assi")
                is_heatmap = (chart_key == "heatmap")
                is_podio   = (chart_key == "podio")
                is_funnel  = (chart_key == "funnel")
                is_radar   = (chart_key == "radar")

                # Etichette dinamiche per heatmap
                if is_heatmap:
                    lbl_x   = "Asse X1 — Colonne"
                    lbl_grp = "Asse X2 — Righe (obbligatorio)"
                elif is_podio:
                    lbl_x   = "Asse X — Nomi (etichette)"
                    lbl_grp = "Dettaglio aggiuntivo (opzionale)"
                elif is_funnel:
                    lbl_x   = "Asse X — Etichette livelli"
                    lbl_grp = "Raggruppa / Colore (opzionale)"
                elif is_radar:
                    lbl_x   = "Asse X — Categorie angoli"
                    lbl_grp = "Raggruppa / Colore (opzionale)"
                else:
                    lbl_x   = "Asse X"
                    lbl_grp = "Raggruppa / Colore (opzionale)"

                # Asse X
                _xi   = visible.index(ss.g3_x_col) if ss.g3_x_col in visible else 0
                x_col = st.selectbox(lbl_x, visible, index=_xi)
                ss.g3_x_col = x_col

                # Asse Y — include Frequenza tranne per grafici che richiedono colonna reale
                _y_no_freq = {"boxplot", "violin", "scatter", "histogram", "radar"}
                if chart_key in _y_no_freq:
                    y_opts  = visible
                    _yi_def = visible.index(ss.g3_y_col) if ss.g3_y_col in visible else 0
                else:
                    y_opts  = visible[:]  # copia
                    # inserisci Frequenza dopo il primo elemento (stringa vuota)
                    y_opts.insert(1, _FREQ_LABEL)
                    _y_ss   = ss.g3_y_col if ss.g3_y_col else ""
                    _yi_def = y_opts.index(_y_ss) if _y_ss in y_opts else 0

                if is_heatmap:
                    lbl_y = "Colore / Valori (opzionale — vuoto = Frequenza)"
                elif is_podio:
                    lbl_y = "Asse Y — Punteggio/Valore (opzionale)"
                elif is_funnel:
                    lbl_y = "Asse Y — Valori (opzionale — vuoto = Frequenza)"
                else:
                    lbl_y = "Asse Y"
                y_col = st.selectbox(lbl_y, y_opts, index=_yi_def)
                ss.g3_y_col = y_col

                # Raggruppa / Asse X2
                _gi     = visible.index(ss.g3_grp_col) if ss.g3_grp_col in visible else 0
                grp_col = st.selectbox(lbl_grp, visible, index=_gi)
                ss.g3_grp_col = grp_col

                # ── Opzioni specifiche per tipo ────────────────────────────────
                if chart_key == "bar":
                    ss.g3_bar_horiz = st.checkbox("Barre orizzontali",
                                                   value=bool(ss.g3_bar_horiz))
                    if not (y_col == _FREQ_LABEL or not y_col):
                        ss.g3_show_line = st.checkbox("Aggiungi linea sovrapposta",
                                                       value=bool(ss.g3_show_line))

                elif chart_key == "bar_stacked":
                    ss.g3_bar_3d = st.checkbox("Variante 3D (Pila mattoncini)",
                                                value=bool(ss.g3_bar_3d))

                elif chart_key == "scatter":
                    ss.g3_show_line = st.checkbox("Mostra linee di collegamento",
                                                   value=bool(ss.g3_show_line))

                elif chart_key == "combo":
                    _lc_opts  = [""] + [c for c in list(df.columns) if c != x_col]
                    _lc_val   = ss.g3_line_col if ss.g3_line_col in _lc_opts else ""
                    _lc_i     = _lc_opts.index(_lc_val) if _lc_val in _lc_opts else 0
                    _sel_lc   = st.selectbox("Colonna per la linea (asse Y2)",
                                              _lc_opts, index=_lc_i)
                    ss.g3_line_col = _sel_lc

                elif chart_key == "podio":
                    ss.g3_podio_n = int(st.number_input(
                        "Numero di posizioni da mostrare", 1, 10,
                        value=int(ss.g3_podio_n), step=1))

            # ── Personalizzazione (legge da session_state, si imposta nella tab ⚙️) ──
            title      = ss.opt_title
            palette_nm = ss.opt_palette_nm
            palette    = PALETTES.get(palette_nm, list(PALETTES.values())[0])
            show_leg   = ss.opt_show_leg
            if not is_map:
                xlabel    = ss.opt_xlabel
                ylabel    = ss.opt_ylabel
                show_grid = ss.opt_show_grid
                show_nums = ss.opt_show_nums
                xtick_rot = ss.opt_xtick_rot
            else:
                xlabel = ylabel = ""
                show_grid = show_nums = False
                xtick_rot = -1

            # ── Filtri ────────────────────────────────────────────────────────
            with st.expander("🔍  Filtri"):
                _nf       = int(st.number_input("Numero di filtri", 0, 8, value=int(ss.g3_n_filt), step=1))
                ss.g3_n_filt = _nf
                df_filt   = df.copy()
                _FC_TYPES = ["Valori (lista)", "Contiene", "Non contiene", "Num ≥", "Num ≤"]
                _fc_opts  = [""] + list(df.columns)

                for fi in range(_nf):
                    _fs = ss.g3_filters[fi]
                    st.markdown(f"**Filtro {fi + 1}**")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        _fci     = _fc_opts.index(_fs["col"]) if _fs["col"] in _fc_opts else 0
                        filt_col = st.selectbox("Colonna", _fc_opts, index=_fci)
                        _fs["col"] = filt_col
                    with fc2:
                        _fti      = _FC_TYPES.index(_fs["type"]) if _fs["type"] in _FC_TYPES else 0
                        filt_type = st.selectbox("Tipo", _FC_TYPES, index=_fti)
                        _fs["type"] = filt_type
                    if filt_col and filt_col in df.columns:
                        if filt_type == "Valori (lista)":
                            uniq  = sorted(df_filt[filt_col].dropna().unique(), key=_xkey)
                            _fdef = [v for v in (_fs.get("val") or []) if v in uniq]
                            sel   = st.multiselect("Valori da includere", uniq, default=_fdef)
                            _fs["val"] = sel
                            if sel:
                                df_filt = df_filt[df_filt[filt_col].astype(str).isin([str(v) for v in sel])]
                        elif filt_type in ("Contiene", "Non contiene"):
                            _fv  = str(_fs.get("val") or "") if not isinstance(_fs.get("val"), list) else ""
                            txt  = st.text_input("Testo", value=_fv)
                            _fs["val"] = txt
                            if txt:
                                mask    = df_filt[filt_col].astype(str).str.contains(txt, case=False, na=False)
                                df_filt = df_filt[mask if filt_type == "Contiene" else ~mask]
                        elif filt_type == "Num ≥":
                            _fv  = float(_fs["val"]) if isinstance(_fs.get("val"), (int, float)) else 0.0
                            val  = st.number_input("Valore minimo", value=_fv)
                            _fs["val"] = val
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") >= val]
                        elif filt_type == "Num ≤":
                            _fv  = float(_fs["val"]) if isinstance(_fs.get("val"), (int, float)) else 0.0
                            val  = st.number_input("Valore massimo", value=_fv)
                            _fs["val"] = val
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") <= val]
                    st.caption(f"→ {len(df_filt):,} righe dopo questo filtro")
                    ss.g3_filters[fi] = _fs

            # ── Separa ────────────────────────────────────────────────────────
            sep_col   = ""
            _SEP_MODES = ["cumulativo", "separa", "valore fisso"]
            with st.expander("✂️  Separa"):
                _ns      = int(st.number_input("Numero di colonne", 0, 6, value=int(ss.g3_n_sep), step=1))
                ss.g3_n_sep = _ns
                _vis_only   = [c for c in visible if c]
                _sv_opts    = [""] + _vis_only
                for _si in range(_ns):
                    _sp = ss.g3_seps[_si]
                    _sca, _scb = st.columns(2)
                    with _sca:
                        _sci = _sv_opts.index(_sp["col"]) if _sp["col"] in _sv_opts else 0
                        _sc  = st.selectbox("Colonna", _sv_opts, index=_sci)
                        _sp["col"] = _sc
                    with _scb:
                        _smi = _SEP_MODES.index(_sp["mode"]) if _sp["mode"] in _SEP_MODES else 0
                        _sm  = st.selectbox("Modalità", _SEP_MODES, index=_smi)
                        _sp["mode"] = _sm
                    if _sc and _sm == "separa" and not sep_col:
                        sep_col = _sc
                    elif _sc and _sm == "valore fisso" and _sc in df_filt.columns:
                        _fv_opts = [""] + sorted([str(v) for v in df_filt[_sc].dropna().unique()], key=_xkey)
                        _fvi     = _fv_opts.index(_sp["fval"]) if _sp["fval"] in _fv_opts else 0
                        _fv      = st.selectbox("Valore fisso", _fv_opts, index=_fvi)
                        _sp["fval"] = _fv
                        if _fv:
                            df_filt = df_filt[df_filt[_sc].astype(str) == _fv]
                    ss.g3_seps[_si] = _sp

            gen = st.button("▶  Genera grafico", type="primary", use_container_width=True)

        # ── Area grafico ──────────────────────────────────────────────────────
        # Raccoglie le opzioni di stile dalla session_state (impostati in tab ⚙️)
        _so = st.session_state
        style_opts = {
            "title_sz":      int(_so.opt_title_sz),
            "title_bold":    bool(_so.opt_title_bold),
            "title_ital":    bool(_so.opt_title_ital),
            "label_sz":      int(_so.opt_label_sz),
            "y_min":         str(_so.opt_y_min),
            "y_max":         str(_so.opt_y_max),
            "minimal":       bool(_so.opt_minimal),
            "show_axes_y":   bool(_so.opt_show_axes_y),
            "white_bg":      bool(_so.opt_white_bg),
            "legend_sz":     int(_so.opt_legend_sz),
        }
        axis_opts = {
            "y_scale":       str(_so.opt_y_scale),
            "x_scale":       str(_so.opt_x_scale),
            "y_step":        str(_so.opt_y_step),
            "x_step":        str(_so.opt_x_step),
            "y_minor":       bool(_so.opt_y_minor),
            "y_minor_ndiv":  int(_so.opt_y_minor_ndiv),
            "x_minor":       bool(_so.opt_x_minor),
            "x_minor_ndiv":  int(_so.opt_x_minor_ndiv),
            "gy_maj":        bool(_so.opt_gy_maj),
            "gy_maj_sty":    str(_so.opt_gy_maj_sty),
            "gy_maj_alp":    float(_so.opt_gy_maj_alp),
            "gy_maj_lw":     float(_so.opt_gy_maj_lw),
            "gy_min":        bool(_so.opt_gy_min),
            "gy_min_sty":    str(_so.opt_gy_min_sty),
            "gy_min_alp":    float(_so.opt_gy_min_alp),
            "gy_min_lw":     float(_so.opt_gy_min_lw),
            "gy_min_ndiv":   int(_so.opt_gy_min_ndiv),
            "gx_maj":        bool(_so.opt_gx_maj),
            "gx_maj_sty":    str(_so.opt_gx_maj_sty),
            "gx_maj_alp":    float(_so.opt_gx_maj_alp),
            "gx_maj_lw":     float(_so.opt_gx_maj_lw),
            "gx_min":        bool(_so.opt_gx_min),
            "gx_min_sty":    str(_so.opt_gx_min_sty),
            "gx_min_alp":    float(_so.opt_gx_min_alp),
            "gx_min_lw":     float(_so.opt_gx_min_lw),
            "gx_min_ndiv":   int(_so.opt_gx_min_ndiv),
            "xtick_sz":      int(_so.opt_xtick_sz),
            "ytick_sz":      int(_so.opt_ytick_sz),
        }

        with right:
            # ── Navigazione Separa ─────────────────────────────────────────────
            _df_chart    = df_filt
            _chart_title = title
            do_render    = gen

            if sep_col and sep_col in df_filt.columns:
                _sv  = sorted(df_filt[sep_col].dropna().unique(), key=_xkey)
                _nsv = len(_sv)
                # Resetta indice quando la colonna separa cambia
                if st.session_state.get("_sep_col_last") != sep_col:
                    st.session_state["_sep_idx"]      = 0
                    st.session_state["_sep_col_last"] = sep_col
                _idx = max(0, min(int(st.session_state.get("_sep_idx", 0)), _nsv - 1))
                st.session_state["_sep_idx"] = _idx

                _nc1, _nc2, _nc3 = st.columns([1, 6, 1])
                with _nc1:
                    if st.button("◀", disabled=(_idx == 0), key="sep_btn_prev"):
                        st.session_state["_sep_idx"] = max(0, _idx - 1)
                        do_render = True
                with _nc2:
                    _cur_v = _sv[st.session_state["_sep_idx"]]
                    st.markdown(
                        f"**{st.session_state['_sep_idx'] + 1} / {_nsv}**"
                        f" — *{sep_col}*: **{_cur_v}**"
                    )
                with _nc3:
                    if st.button("▶", disabled=(_idx >= _nsv - 1), key="sep_btn_next"):
                        st.session_state["_sep_idx"] = min(_nsv - 1, _idx + 1)
                        do_render = True

                _cur_v       = _sv[st.session_state["_sep_idx"]]
                _df_chart    = df_filt[df_filt[sep_col].astype(str) == str(_cur_v)]
                _chart_title = title or f"{sep_col}: {_cur_v}"

            if do_render:
                try:
                    _fw = float(st.session_state.opt_fig_w)
                    _fh = float(st.session_state.opt_fig_h)

                    if not is_map and palette:
                        plt.rcParams["axes.prop_cycle"] = plt.cycler(
                            color=_colors(10, palette)
                        )

                    # ── MAPPE ─────────────────────────────────────────────────
                    if is_map:
                        if not GEO_OK:
                            raise ImportError("Installa geopandas per usare le mappe.")
                        if not geo_col:
                            raise ValueError("Seleziona la colonna geo.")

                        fig, ax = plt.subplots(figsize=(_fw, _fh))
                        gdf = _load_geo(scope_key)
                        if gdf is None:
                            raise ValueError(
                                f"Impossibile caricare le geometrie per '{scope_label}'."
                            )

                        if chart_key == "choropleth":
                            if not val_col_map:
                                raise ValueError(
                                    "Seleziona il Valore numerico per la mappa coropletica."
                                )
                            _plot_choropleth(fig, ax, _df_chart, geo_col, val_col_map,
                                             gdf, palette, _chart_title)

                        elif chart_key == "bubble_map":
                            _plot_bubble_map(fig, ax, _df_chart, geo_col,
                                             val_col_map or None,
                                             grp_col_map or None,
                                             gdf, palette, _chart_title, show_leg)

                        elif chart_key == "band_map":
                            _orient = "Verticale" if band_orient.startswith("V") else "Orizzontale"
                            _plot_band_map(fig, ax, _df_chart, geo_col,
                                           grp_col_map or None,
                                           val_col_map or None,
                                           gdf, palette, _chart_title, show_leg,
                                           orientation=_orient,
                                           equal_area=band_equal_area)

                        elif chart_key == "pin_map":
                            _plot_pin_map(fig, ax, _df_chart, geo_col,
                                          val_col_map or None,
                                          grp_col_map or None,
                                          gdf, palette, _chart_title, show_leg)

                    # ── GRAFICI STANDARD ──────────────────────────────────────
                    else:
                        # y_col può essere _FREQ_LABEL (stringa speciale) o colonna reale
                        x    = x_col or None
                        y    = y_col if y_col else None
                        g    = grp_col or None
                        _dlz = int(_so.opt_data_lbl_sz)

                        fig, ax = plt.subplots(figsize=(_fw, _fh))
                        ax = _render_chart_on_ax(
                            ax, _df_chart, chart_key, x, y, g,
                            palette, show_nums, xtick_rot, _dlz,
                            extra={
                                "horiz":      bool(ss.g3_bar_horiz),
                                "stacked_3d": bool(ss.g3_bar_3d),
                                "show_line":  bool(ss.g3_show_line),
                                "line_col":   ss.g3_line_col or None,
                                "podio_n":    int(ss.g3_podio_n),
                            },
                        )
                        # non applicare impostazioni asse su grafici con layout custom
                        _is_3d = (chart_key == "bar_stacked" and bool(ss.g3_bar_3d))
                        if chart_key not in _NO_AXIS_KEYS and not _is_3d:
                            _apply_style(ax, _chart_title, xlabel, ylabel,
                                         show_leg, show_grid, opts=style_opts)
                            _apply_axis_settings(ax, axis_opts)
                        else:
                            if _chart_title:
                                ax.set_title(_chart_title, fontsize=13, fontweight="bold")

                    fig.tight_layout(pad=1.5)
                    st.session_state.fig = fig

                except Exception as e:
                    st.error(f"Errore grafico: {e}")
                    st.session_state.fig = None

            # ── Mostra grafico + download ──────────────────────────────────────
            if st.session_state.fig is not None:
                st.pyplot(st.session_state.fig, use_container_width=True)

                buf = io.BytesIO()
                dpi_out = int(st.session_state.opt_dpi)
                st.session_state.fig.savefig(buf, format="png", dpi=dpi_out, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label="💾  Scarica PNG",
                    data=buf,
                    file_name="grafico.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.info("Imposta le opzioni a sinistra e clicca **▶ Genera grafico**.")

        st.divider()
        if st.button("⚙️  Vai alle Impostazioni →", use_container_width=True):
            st.session_state.active_tab = 3
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGINA 4 — PERSONALIZZAZIONE
# Usa value= + writeback immediato (NO key=) per evitare che Streamlit
# cancelli i valori quando si torna su un'altra tab.
# ══════════════════════════════════════════════════════════════════════════════
elif _sel == _TABS[3]:
    ss = st.session_state
    st.subheader("⚙️  Personalizzazione grafico")
    st.caption("Le impostazioni si applicano al prossimo **▶ Genera grafico**.")

    # ── 🎨 Stile ──────────────────────────────────────────────────────────────
    with st.expander("🎨  Stile e colori", expanded=True):
        _pal_keys = list(PALETTES.keys())
        _pal_idx  = _pal_keys.index(ss.opt_palette_nm) if ss.opt_palette_nm in _pal_keys else 0
        ss.opt_palette_nm = st.selectbox("Palette colori", _pal_keys, index=_pal_idx)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            ss.opt_show_leg    = st.checkbox("Mostra legenda",        value=ss.opt_show_leg)
        with sc2:
            ss.opt_show_grid   = st.checkbox("Griglia",               value=ss.opt_show_grid)
        with sc3:
            ss.opt_show_nums   = st.checkbox("Numeri sulle barre",    value=ss.opt_show_nums)

        sc4, sc5, sc6 = st.columns(3)
        with sc4:
            ss.opt_minimal     = st.checkbox("Stile minimal",         value=ss.opt_minimal,
                                             help="Rimuove le spine superiore e destra del grafico")
        with sc5:
            ss.opt_show_axes_y = st.checkbox("Mostra asse Y",         value=ss.opt_show_axes_y)
        with sc6:
            ss.opt_white_bg    = st.checkbox("Sfondo bianco",         value=ss.opt_white_bg)

    # ── 📝 Testi ──────────────────────────────────────────────────────────────
    with st.expander("📝  Titolo ed etichette assi", expanded=True):
        ss.opt_title = st.text_input("Titolo grafico", value=ss.opt_title)
        tc1, tc2, tc3 = st.columns([2, 1, 1])
        with tc1:
            ss.opt_title_sz   = int(st.number_input("Dim. titolo", min_value=6, max_value=36,
                                                     value=int(ss.opt_title_sz), step=1))
        with tc2:
            ss.opt_title_bold = st.checkbox("Grassetto", value=ss.opt_title_bold, key="_tb")
        with tc3:
            ss.opt_title_ital = st.checkbox("Corsivo",   value=ss.opt_title_ital, key="_ti")

        st.markdown("---")
        xa1, xa2 = st.columns(2)
        with xa1:
            ss.opt_xlabel = st.text_input("Etichetta Asse X", value=ss.opt_xlabel)
        with xa2:
            ss.opt_ylabel = st.text_input("Etichetta Asse Y", value=ss.opt_ylabel)

        fsz1, fsz2, fsz3, fsz4 = st.columns(4)
        with fsz1:
            ss.opt_label_sz  = int(st.number_input("Dim. etich. assi", min_value=6, max_value=24,
                                                    value=int(ss.opt_label_sz), step=1))
        with fsz2:
            ss.opt_xtick_sz  = int(st.number_input("Dim. tick X", min_value=4, max_value=20,
                                                    value=int(ss.opt_xtick_sz), step=1))
        with fsz3:
            ss.opt_ytick_sz  = int(st.number_input("Dim. tick Y", min_value=4, max_value=20,
                                                    value=int(ss.opt_ytick_sz), step=1))
        with fsz4:
            ss.opt_legend_sz = int(st.number_input("Dim. legenda", min_value=4, max_value=20,
                                                    value=int(ss.opt_legend_sz), step=1))

    # ── 📐 Assi e scala ───────────────────────────────────────────────────────
    with st.expander("📐  Assi e scala", expanded=False):
        # Scala
        _sc_keys = list(_SCALE_TYPES)
        sca1, sca2 = st.columns(2)
        with sca1:
            _yi = _sc_keys.index(ss.opt_y_scale) if ss.opt_y_scale in _sc_keys else 0
            ss.opt_y_scale = st.selectbox("Scala Asse Y", _sc_keys, index=_yi)
        with sca2:
            _xi = _sc_keys.index(ss.opt_x_scale) if ss.opt_x_scale in _sc_keys else 0
            ss.opt_x_scale = st.selectbox("Scala Asse X", _sc_keys, index=_xi)

        # Limiti e passo
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            ss.opt_y_min  = st.text_input("Y min", value=ss.opt_y_min)
        with lc2:
            ss.opt_y_max  = st.text_input("Y max", value=ss.opt_y_max)
        with lc3:
            ss.opt_y_step = st.text_input("Y passo tick", value=ss.opt_y_step,
                                           help="Lascia vuoto per automatico")
        with lc4:
            ss.opt_x_step = st.text_input("X passo tick", value=ss.opt_x_step,
                                           help="Lascia vuoto per automatico")

        # Minor tick
        st.markdown("**Minor tick**")
        mt1, mt2, mt3, mt4 = st.columns(4)
        with mt1:
            ss.opt_y_minor = st.checkbox("Minor tick Y", value=ss.opt_y_minor)
        with mt2:
            ss.opt_y_minor_ndiv = int(st.number_input("Subdiv Y", min_value=2, max_value=20,
                                                        value=int(ss.opt_y_minor_ndiv), step=1))
        with mt3:
            ss.opt_x_minor = st.checkbox("Minor tick X", value=ss.opt_x_minor)
        with mt4:
            ss.opt_x_minor_ndiv = int(st.number_input("Subdiv X", min_value=2, max_value=20,
                                                        value=int(ss.opt_x_minor_ndiv), step=1))

        # Rotazione etichette X
        ss.opt_xtick_rot = int(st.slider(
            "Rotazione etichette X  (−1 = auto)",
            min_value=-1, max_value=90, value=int(ss.opt_xtick_rot), step=5,
        ))

    # ── 🔲 Griglia avanzata ────────────────────────────────────────────────────
    with st.expander("🔲  Griglia avanzata", expanded=False):
        st.caption("Se abilitata, sovrascrive l'opzione 'Griglia' dello stile.")
        _gs_opts = list(_GRID_STYLE_MAP.keys())

        st.markdown("**Asse Y — major**")
        gy1, gy2, gy3, gy4 = st.columns([1, 2, 1, 1])
        with gy1:
            ss.opt_gy_maj = st.checkbox("Attiva##gym", value=ss.opt_gy_maj)
        with gy2:
            _gi = _gs_opts.index(ss.opt_gy_maj_sty) if ss.opt_gy_maj_sty in _gs_opts else 1
            ss.opt_gy_maj_sty = st.selectbox("Stile##gym", _gs_opts, index=_gi)
        with gy3:
            ss.opt_gy_maj_alp = float(st.number_input("Alpha##gym", 0.05, 1.0,
                                                        value=float(ss.opt_gy_maj_alp), step=0.05))
        with gy4:
            ss.opt_gy_maj_lw  = float(st.number_input("Spess.##gym", 0.2, 3.0,
                                                        value=float(ss.opt_gy_maj_lw), step=0.2))

        st.markdown("**Asse Y — minor**")
        gy5, gy6, gy7, gy8, gy9 = st.columns([1, 2, 1, 1, 1])
        with gy5:
            ss.opt_gy_min = st.checkbox("Attiva##gymi", value=ss.opt_gy_min)
        with gy6:
            _gi2 = _gs_opts.index(ss.opt_gy_min_sty) if ss.opt_gy_min_sty in _gs_opts else 2
            ss.opt_gy_min_sty = st.selectbox("Stile##gymi", _gs_opts, index=_gi2)
        with gy7:
            ss.opt_gy_min_alp = float(st.number_input("Alpha##gymi", 0.05, 1.0,
                                                        value=float(ss.opt_gy_min_alp), step=0.05))
        with gy8:
            ss.opt_gy_min_lw  = float(st.number_input("Spess.##gymi", 0.2, 3.0,
                                                        value=float(ss.opt_gy_min_lw), step=0.2))
        with gy9:
            ss.opt_gy_min_ndiv = int(st.number_input("Subdiv##gymi", 2, 20,
                                                       value=int(ss.opt_gy_min_ndiv), step=1))

        st.markdown("**Asse X — major**")
        gx1, gx2, gx3, gx4 = st.columns([1, 2, 1, 1])
        with gx1:
            ss.opt_gx_maj = st.checkbox("Attiva##gxm", value=ss.opt_gx_maj)
        with gx2:
            _gi3 = _gs_opts.index(ss.opt_gx_maj_sty) if ss.opt_gx_maj_sty in _gs_opts else 1
            ss.opt_gx_maj_sty = st.selectbox("Stile##gxm", _gs_opts, index=_gi3)
        with gx3:
            ss.opt_gx_maj_alp = float(st.number_input("Alpha##gxm", 0.05, 1.0,
                                                        value=float(ss.opt_gx_maj_alp), step=0.05))
        with gx4:
            ss.opt_gx_maj_lw  = float(st.number_input("Spess.##gxm", 0.2, 3.0,
                                                        value=float(ss.opt_gx_maj_lw), step=0.2))

        st.markdown("**Asse X — minor**")
        gx5, gx6, gx7, gx8, gx9 = st.columns([1, 2, 1, 1, 1])
        with gx5:
            ss.opt_gx_min = st.checkbox("Attiva##gxmi", value=ss.opt_gx_min)
        with gx6:
            _gi4 = _gs_opts.index(ss.opt_gx_min_sty) if ss.opt_gx_min_sty in _gs_opts else 2
            ss.opt_gx_min_sty = st.selectbox("Stile##gxmi", _gs_opts, index=_gi4)
        with gx7:
            ss.opt_gx_min_alp = float(st.number_input("Alpha##gxmi", 0.05, 1.0,
                                                        value=float(ss.opt_gx_min_alp), step=0.05))
        with gx8:
            ss.opt_gx_min_lw  = float(st.number_input("Spess.##gxmi", 0.2, 3.0,
                                                        value=float(ss.opt_gx_min_lw), step=0.2))
        with gx9:
            ss.opt_gx_min_ndiv = int(st.number_input("Subdiv##gxmi", 2, 20,
                                                       value=int(ss.opt_gx_min_ndiv), step=1))

    # ── 🔤 Etichette dati ────────────────────────────────────────────────────
    with st.expander("🔤  Etichette dati (numeri su barre)", expanded=False):
        ss.opt_data_lbl_sz = int(st.number_input(
            "Grandezza font etichette", min_value=4, max_value=20,
            value=int(ss.opt_data_lbl_sz), step=1,
        ))

    # ── 💾 Esportazione ───────────────────────────────────────────────────────
    with st.expander("💾  Esportazione PNG", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            ss.opt_fig_w = float(st.number_input("Larghezza (pollici)", min_value=4.0,
                                                   max_value=24.0, value=float(ss.opt_fig_w), step=0.5))
        with ec2:
            ss.opt_fig_h = float(st.number_input("Altezza (pollici)", min_value=3.0,
                                                   max_value=20.0, value=float(ss.opt_fig_h), step=0.5))
        with ec3:
            ss.opt_dpi   = int(st.number_input("DPI", min_value=72, max_value=600,
                                                value=int(ss.opt_dpi), step=50))

    st.divider()
    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("🔄  Ripristina default", use_container_width=True):
            for _k, _v in _OPT_DEFAULTS.items():
                st.session_state[_k] = _v
            st.rerun()
    with rc2:
        if st.button("📊  Vai al Grafico →", use_container_width=True):
            st.session_state.active_tab = 2
            st.rerun()
