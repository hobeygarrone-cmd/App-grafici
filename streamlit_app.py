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
    ("Barre verticali",    "bar_v"),
    ("Barre orizzontali",  "bar_h"),
    ("Barre stacked",      "bar_stacked"),
    ("Linee",              "line"),
    ("Dispersione (XY)",   "scatter"),
    ("Istogramma",         "histogram"),
    ("Torta",              "pie"),
    ("Heatmap",            "heatmap"),
    ("Boxplot",            "boxplot"),
    ("Mappa coropletica",  "choropleth"),
    ("Mappa bolla",        "bubble_map"),
    ("Mappa bande",        "band_map"),
    ("Mappa pin",          "pin_map"),
]

MAP_KEYS = {"choropleth", "bubble_map", "band_map", "pin_map"}

_SDG_COLORS = [
    "#E5243B","#DDA63A","#4C9F38","#C5192D","#FF3A21",
    "#26BDE2","#FCC30B","#A21942","#FD6925","#DD1367",
    "#FD9D24","#BF8B2E","#3F7E44","#0A97D9","#56C02B",
    "#00689D","#19486A",
]

# Palette istituzionali (liste di colori esadecimali)
_BRAND_PALETTES = {
    "CRT":     ["#312A74","#FDC400","#5C5499","#F0A800","#1E1A55","#FFD966","#7B73AA","#E8C000"],
    "ISP":     ["#004258","#B10931","#326779","#D43060","#005A74","#E05070","#007A9A","#FF6080"],
    "Cariplo": ["#002F6C","#E31937","#0057B8","#FF4D4D","#003F8A","#C0392B","#1565C0","#FF7043"],
    "Carisbo": ["#1C1C1C","#D40000","#555555","#FF6666","#333333","#B30000","#888888","#FF9999"],
    "BS2026":  ["#1E3264","#3D4FA0","#9089C0","#2A6B5A","#3B9878","#7BC5A0","#F0A020","#888FA0"],
    "BS26-2":  ["#21345C","#5B8DC8","#3EA8A5","#D98A25","#8870B0","#89C0A5","#A89880","#C0CEDE"],
    "BS26-3":  ["#1C2B5A","#3C68B5","#E8A020","#8890A0","#5878C8","#C47A10","#2A3870","#B0B8C8"],
    "Default": ["#156082","#E97132","#196B24","#0F9ED5","#A02B93","#4EA72E","#467886","#96607D"],
    "SDG":     _SDG_COLORS,
}

PALETTES = {
    # ── Istituzionali ────────────────────────────────────────────────────────
    "CRT":       "CRT",
    "ISP":       "ISP",
    "Cariplo":   "Cariplo",
    "Carisbo":   "Carisbo",
    "BS2026":    "BS2026",
    "BS26-2":    "BS26-2",
    "BS26-3":    "BS26-3",
    "Default":   "Default",
    "SDG":       "SDG",
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


def _agg(df_, x_col, y_col, grp_col=None):
    g_cols = [c for c in [x_col, grp_col] if c]
    if y_col and g_cols:
        return df_.groupby(g_cols, sort=False)[y_col].sum().reset_index()
    if x_col:
        return df_[x_col].value_counts().reset_index().rename(
            columns={x_col: x_col, "count": "_n"})
    return df_


def _colors(n, palette_name):
    brand = _BRAND_PALETTES.get(palette_name)
    if brand:
        return [brand[i % len(brand)] for i in range(n)]
    cmap = plt.get_cmap(palette_name) if palette_name else plt.get_cmap("tab10")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _cat_colors(categories, palette_name):
    cats = list(categories)
    cols = _colors(len(cats), palette_name)
    return dict(zip(cats, cols))


def _apply_style(ax_, title, xlabel, ylabel, show_legend, show_grid, fontsize_title=13):
    if title:
        ax_.set_title(title, fontsize=fontsize_title, fontweight="bold", pad=10)
    if xlabel:
        ax_.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax_.set_ylabel(ylabel, fontsize=10)
    if show_grid:
        ax_.grid(True, alpha=0.3, linestyle="--", axis="y")
    if show_legend:
        handles, labs = ax_.get_legend_handles_labels()
        if handles:
            ax_.legend(fontsize=8, framealpha=0.85)


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

    cmap_name = palette or "Blues"
    vmin = agg["value"].min()
    vmax = agg["value"].max()
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
        vmin, vmax = 0, 1

    merged.plot(
        column="value", ax=ax, cmap=cmap_name,
        missing_kwds={"color": "#e4e4e4", "label": "Nessun dato"},
        legend=False, edgecolor="#aaaaaa", linewidth=0.5,
        vmin=vmin, vmax=vmax,
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = mcm.ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
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
        groups  = sorted(agg[grp_col].dropna().unique(), key=str)
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
        base_color = plt.get_cmap(palette or "tab10")(0)
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

    cats = sorted(df_filt[grp_col].dropna().unique(), key=str)
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
        groups = sorted(df2[grp_col].dropna().unique(), key=str)
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
        base_color = plt.get_cmap(palette or "tab10")(0.2)
        if val_col and val_col in df2.columns:
            agg = df2.groupby("_GEO")[val_col].sum().reset_index()
            vvals = pd.to_numeric(agg[val_col], errors="coerce")
            vmin, vmax = vvals.min(), vvals.max()
            cmap = plt.get_cmap(palette or "YlOrRd")
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


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGAZIONE  (radio orizzontale — permette st.rerun() per cambiare pagina)
# ══════════════════════════════════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

_TABS = ["📁  Dati", "🗂  Variabili", "📊  Grafico"]
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
            st.session_state.col_roles = {
                c: ("dipend" if _is_num(df_new[c]) else "indip")
                for c in df_new.columns
            }
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
        st.subheader("Ruolo di ogni colonna")
        st.caption(
            "**indip** = variabile indipendente (Asse X, categorie)  |  "
            "**dipend** = variabile dipendente (Asse Y, valori)  |  "
            "**ignora** = escludi dal grafico"
        )

        roles        = st.session_state.col_roles.copy()
        ROLE_OPTIONS = ["indip", "dipend", "ignora"]

        # ── Azioni rapide ──────────────────────────────────────────────────────
        ca, cb, cc_ = st.columns([1, 2, 1])
        with ca:
            if st.button("Auto-rileva", use_container_width=True,
                         help="Assegna 'dipend' alle colonne numeriche, 'indip' alle altre"):
                for col in df.columns:
                    roles[col] = "dipend" if _is_num(df[col]) else "indip"
                st.session_state.col_roles = roles
                st.rerun()
        with cb:
            bulk = st.selectbox(
                "Assegna a tutti:", ["", "indip", "dipend", "ignora"],
                key="bulk_role", label_visibility="collapsed",
            )
        with cc_:
            if st.button("Applica", use_container_width=True) and bulk:
                for col in df.columns:
                    roles[col] = bulk
                st.session_state.col_roles = roles
                st.rerun()

        st.divider()

        # ── Selettori per colonna ──────────────────────────────────────────────
        N_COLS   = 4
        col_list = list(df.columns)
        for row_start in range(0, len(col_list), N_COLS):
            row_widgets = st.columns(N_COLS)
            for j, col_name in enumerate(col_list[row_start : row_start + N_COLS]):
                with row_widgets[j]:
                    cur      = roles.get(col_name, "indip")
                    idx      = ROLE_OPTIONS.index(cur) if cur in ROLE_OPTIONS else 0
                    tipo     = "numerico" if _is_num(df[col_name]) else "testo"
                    new_role = st.selectbox(
                        col_name, ROLE_OPTIONS, index=idx,
                        key=f"role_{col_name}", help=f"Tipo: {tipo}",
                    )
                    roles[col_name] = new_role

        st.session_state.col_roles = roles

        show_cols = [c for c, r in roles.items() if r != "ignora"]
        if show_cols:
            st.divider()
            st.subheader("Anteprima dati")
            st.data_editor(df[show_cols].head(300), use_container_width=True, key="preview_editor")

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
        indip   = [""] + [c for c, r in roles.items() if r == "indip"]
        dipend  = [""] + [c for c, r in roles.items() if r == "dipend"]
        all_col = [""] + [c for c, r in roles.items() if r != "ignora"]

        left, right = st.columns([1, 2])

        with left:
            # ── Tipo grafico ──────────────────────────────────────────────────
            st.subheader("Tipo di grafico")
            chart_labels = [t[0] for t in CHART_TYPES]
            chart_keys   = [t[1] for t in CHART_TYPES]
            chart_label  = st.selectbox("", chart_labels, label_visibility="collapsed")
            chart_key    = chart_keys[chart_labels.index(chart_label)]
            is_map       = chart_key in MAP_KEYS

            # ── Opzioni mappa (solo per grafici mappa) ────────────────────────
            if is_map:
                st.subheader("Impostazioni mappa")
                if not GEO_OK:
                    st.error("geopandas non installato — le mappe non sono disponibili.")
                scope_label = st.selectbox("Ambito mappa", [s[0] for s in MAP_SCOPES])
                scope_key   = MAP_SCOPE_KEYS[scope_label]
                geo_col     = st.selectbox(
                    "Colonna geo (nomi geografici nel tuo file)",
                    [""] + list(df.columns),
                    help="Seleziona la colonna che contiene i nomi delle aree geografiche"
                )
                if chart_key == "band_map":
                    val_col_map = st.selectbox("Valore (opzionale, per proporzionare bande)", dipend)
                    grp_col_map = st.selectbox("Categoria colore (obbligatorio)", all_col)
                    band_orient = st.radio(
                        "Direzione bande",
                        ["Verticale (→)", "Orizzontale (↑)"],
                        horizontal=True,
                    )
                    band_equal_area = st.checkbox(
                        "Area proporzionale al valore",
                        value=False,
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
                x_col   = st.selectbox("Asse X (indip / categorie)", indip)
                y_col   = st.selectbox("Asse Y (dipend / valori)", dipend)
                grp_col = st.selectbox("Raggruppa / Colore (opzionale)", all_col)

            # ── Personalizzazione ─────────────────────────────────────────────
            with st.expander("🎨  Personalizzazione", expanded=True):
                title      = st.text_input("Titolo grafico")
                palette_nm = st.selectbox("Palette colori", list(PALETTES.keys()))
                palette    = PALETTES[palette_nm]
                show_leg   = st.checkbox("Mostra legenda", value=True)
                if not is_map:
                    xlabel    = st.text_input("Etichetta Asse X")
                    ylabel    = st.text_input("Etichetta Asse Y")
                    show_grid = st.checkbox("Griglia", value=True)
                    show_nums = st.checkbox("Numeri sulle barre", value=False)
                    xtick_rot = st.slider(
                        "Rotazione etichette X  (−1 = auto)",
                        min_value=-1, max_value=90, value=-1, step=5,
                    )
                else:
                    xlabel = ylabel = ""
                    show_grid = show_nums = False
                    xtick_rot = -1

            # ── Filtri ────────────────────────────────────────────────────────
            with st.expander("🔍  Filtri"):
                n_filters = st.number_input("Numero di filtri", 0, 8, 0, step=1)
                df_filt   = df.copy()

                for fi in range(int(n_filters)):
                    st.markdown(f"**Filtro {fi + 1}**")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        filt_col = st.selectbox("Colonna", [""] + list(df.columns), key=f"fc_{fi}")
                    with fc2:
                        filt_type = st.selectbox(
                            "Tipo",
                            ["Valori (lista)", "Contiene", "Non contiene", "Num ≥", "Num ≤"],
                            key=f"ft_{fi}",
                        )
                    if filt_col and filt_col in df.columns:
                        if filt_type == "Valori (lista)":
                            uniq = sorted(df_filt[filt_col].dropna().unique(), key=str)
                            sel  = st.multiselect("Valori da includere", uniq, key=f"fv_{fi}")
                            if sel:
                                df_filt = df_filt[df_filt[filt_col].astype(str).isin([str(v) for v in sel])]
                        elif filt_type in ("Contiene", "Non contiene"):
                            txt = st.text_input("Testo", key=f"fp_{fi}")
                            if txt:
                                mask = df_filt[filt_col].astype(str).str.contains(txt, case=False, na=False)
                                df_filt = df_filt[mask if filt_type == "Contiene" else ~mask]
                        elif filt_type == "Num ≥":
                            val = st.number_input("Valore minimo", key=f"fp_{fi}", value=0.0)
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") >= val]
                        elif filt_type == "Num ≤":
                            val = st.number_input("Valore massimo", key=f"fp_{fi}", value=0.0)
                            df_filt = df_filt[pd.to_numeric(df_filt[filt_col], errors="coerce") <= val]
                    st.caption(f"→ {len(df_filt):,} righe dopo questo filtro")

            gen = st.button("▶  Genera grafico", type="primary", use_container_width=True)

        # ── Area grafico ──────────────────────────────────────────────────────
        with right:
            if gen:
                try:
                    # Dimensione figura
                    figsize = (11, 8) if is_map else (11, 6)
                    fig, ax = plt.subplots(figsize=figsize)

                    if not is_map and palette:
                        plt.rcParams["axes.prop_cycle"] = plt.cycler(
                            color=[plt.get_cmap(palette)(i / 9) for i in range(10)]
                        )

                    # ── MAPPE ─────────────────────────────────────────────────
                    if is_map:
                        if not GEO_OK:
                            raise ImportError("Installa geopandas per usare le mappe.")
                        if not geo_col:
                            raise ValueError("Seleziona la colonna geo.")

                        gdf = _load_geo(scope_key)
                        if gdf is None:
                            raise ValueError(f"Impossibile caricare le geometrie per '{scope_label}'.")

                        if chart_key == "choropleth":
                            if not val_col_map:
                                raise ValueError("Seleziona il Valore numerico per la mappa coropletica.")
                            _plot_choropleth(fig, ax, df_filt, geo_col, val_col_map,
                                             gdf, palette, title)

                        elif chart_key == "bubble_map":
                            _plot_bubble_map(fig, ax, df_filt, geo_col,
                                             val_col_map or None,
                                             grp_col_map or None,
                                             gdf, palette, title, show_leg)

                        elif chart_key == "band_map":
                            _orient = "Verticale" if band_orient.startswith("V") else "Orizzontale"
                            _plot_band_map(fig, ax, df_filt, geo_col,
                                           grp_col_map or None,
                                           val_col_map or None,
                                           gdf, palette, title, show_leg,
                                           orientation=_orient,
                                           equal_area=band_equal_area)

                        elif chart_key == "pin_map":
                            _plot_pin_map(fig, ax, df_filt, geo_col,
                                          val_col_map or None,
                                          grp_col_map or None,
                                          gdf, palette, title, show_leg)

                    # ── GRAFICI STANDARD ──────────────────────────────────────
                    else:
                        x = x_col or None
                        y = y_col or None
                        g = grp_col or None

                        if chart_key == "bar_v":
                            if x and y and g:
                                agg    = _agg(df_filt, x, y, g)
                                pivot  = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
                                colors = _colors(len(pivot.columns), palette)
                                pivot.plot.bar(ax=ax, color=colors)
                                _smart_labels(ax, pivot.index, xtick_rot)
                                if show_nums:
                                    for container in ax.containers:
                                        ax.bar_label(container, fmt="%.4g", fontsize=7, padding=2)
                            elif x and y:
                                agg    = _agg(df_filt, x, y)
                                vals   = pd.to_numeric(agg[y], errors="coerce").fillna(0)
                                colors = _colors(len(agg), palette)
                                ax.bar(range(len(agg)), vals, color=colors)
                                ax.set_xticks(range(len(agg)))
                                _smart_labels(ax, agg[x].astype(str), xtick_rot)
                                if show_nums:
                                    ax.bar_label(ax.containers[0], fmt="%.4g", fontsize=7, padding=2)
                            elif x:
                                vc     = df_filt[x].value_counts()
                                colors = _colors(len(vc), palette)
                                ax.bar(range(len(vc)), vc.values, color=colors)
                                ax.set_xticks(range(len(vc)))
                                _smart_labels(ax, vc.index.astype(str), xtick_rot)
                                if show_nums:
                                    ax.bar_label(ax.containers[0], fmt="%d", fontsize=7, padding=2)
                            else:
                                raise ValueError("Seleziona almeno Asse X.")

                        elif chart_key == "bar_h":
                            if x and y and g:
                                agg   = _agg(df_filt, x, y, g)
                                pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
                                pivot.plot.barh(ax=ax, color=_colors(len(pivot.columns), palette))
                            elif x and y:
                                agg  = _agg(df_filt, x, y)
                                vals = pd.to_numeric(agg[y], errors="coerce").fillna(0)
                                ax.barh(range(len(agg)), vals, color=_colors(len(agg), palette))
                                ax.set_yticks(range(len(agg)))
                                ax.set_yticklabels(agg[x].astype(str), fontsize=7)
                            elif x:
                                vc = df_filt[x].value_counts()
                                ax.barh(range(len(vc)), vc.values, color=_colors(len(vc), palette))
                                ax.set_yticks(range(len(vc)))
                                ax.set_yticklabels(vc.index.astype(str), fontsize=7)
                            else:
                                raise ValueError("Seleziona almeno Asse X.")

                        elif chart_key == "bar_stacked":
                            if not (x and y and g):
                                raise ValueError("Barre stacked richiedono Asse X, Asse Y e Raggruppa.")
                            agg   = _agg(df_filt, x, y, g)
                            pivot = agg.pivot_table(index=x, columns=g, values=y, aggfunc="sum").fillna(0)
                            pivot.plot.bar(ax=ax, stacked=True, color=_colors(len(pivot.columns), palette))
                            _smart_labels(ax, pivot.index, xtick_rot)

                        elif chart_key == "line":
                            if not (x and y):
                                raise ValueError("Seleziona Asse X e Asse Y.")
                            x_all = sorted(df_filt[x].dropna().unique(), key=str)
                            x_pos = {str(v): i for i, v in enumerate(x_all)}
                            if g:
                                groups = sorted(df_filt[g].dropna().unique(), key=str)
                                colors = _colors(len(groups), palette)
                                for grp, col_ in zip(groups, colors):
                                    sub   = df_filt[df_filt[g] == grp].sort_values(x)
                                    xvals = [x_pos.get(str(v)) for v in sub[x]]
                                    yvals = pd.to_numeric(sub[y], errors="coerce").tolist()
                                    pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                                             if xv is not None and not pd.isna(yv)]
                                    if pairs:
                                        xs_, ys_ = zip(*pairs)
                                        ax.plot(list(xs_), list(ys_), marker="o", markersize=4,
                                                label=str(grp), color=col_)
                            else:
                                agg   = df_filt.groupby(x, sort=False)[y].sum().reset_index()
                                xvals = [x_pos.get(str(v)) for v in agg[x]]
                                yvals = pd.to_numeric(agg[y], errors="coerce").tolist()
                                pairs = [(xv, yv) for xv, yv in zip(xvals, yvals)
                                         if xv is not None and not pd.isna(yv)]
                                if pairs:
                                    xs_, ys_ = zip(*pairs)
                                    ax.plot(list(xs_), list(ys_), marker="o", markersize=4)
                            ax.set_xticks(range(len(x_all)))
                            _smart_labels(ax, [str(v) for v in x_all], xtick_rot)

                        elif chart_key == "scatter":
                            if not (x and y):
                                raise ValueError("Seleziona Asse X e Asse Y.")
                            if g:
                                groups = sorted(df_filt[g].dropna().unique(), key=str)
                                colors = _colors(len(groups), palette)
                                for grp, col_ in zip(groups, colors):
                                    sub = df_filt[df_filt[g] == grp]
                                    ax.scatter(pd.to_numeric(sub[x], errors="coerce"),
                                               pd.to_numeric(sub[y], errors="coerce"),
                                               label=str(grp), alpha=0.7, s=30, color=col_)
                            else:
                                ax.scatter(pd.to_numeric(df_filt[x], errors="coerce"),
                                           pd.to_numeric(df_filt[y], errors="coerce"),
                                           alpha=0.7, s=30,
                                           c=[plt.get_cmap(palette or "tab10")(0)])

                        elif chart_key == "histogram":
                            col_ = y or x
                            if not col_:
                                raise ValueError("Seleziona una colonna numerica.")
                            if g:
                                groups = sorted(df_filt[g].dropna().unique(), key=str)
                                colors = _colors(len(groups), palette)
                                for grp, col_c in zip(groups, colors):
                                    vals = pd.to_numeric(
                                        df_filt.loc[df_filt[g] == grp, col_], errors="coerce"
                                    ).dropna()
                                    ax.hist(vals, bins=20, alpha=0.55, label=str(grp), color=col_c)
                            else:
                                vals = pd.to_numeric(df_filt[col_], errors="coerce").dropna()
                                ax.hist(vals, bins=20, color=plt.get_cmap(palette or "tab10")(0))

                        elif chart_key == "pie":
                            if y and x:
                                vc = df_filt.groupby(x)[y].sum().sort_values(ascending=False)
                            elif x:
                                vc = df_filt[x].value_counts()
                            elif y:
                                vc = df_filt[y].value_counts()
                            else:
                                raise ValueError("Seleziona almeno una colonna.")
                            top = vc.head(12)
                            wedges, texts, autotexts = ax.pie(
                                top.values,
                                labels=[str(l) for l in top.index],
                                autopct="%1.1f%%", startangle=90,
                                colors=_colors(len(top), palette),
                            )
                            for t in autotexts:
                                t.set_fontsize(8)
                            ax.axis("equal")

                        elif chart_key == "heatmap":
                            if not (x and y):
                                raise ValueError("Seleziona Asse X (righe) e Asse Y (colonne).")
                            val_col = g or next(
                                (c for c, r in roles.items() if r == "dipend" and c != y), None)
                            if val_col:
                                piv = df_filt.pivot_table(index=x, columns=y, values=val_col, aggfunc="mean")
                            else:
                                piv = df_filt.pivot_table(index=x, columns=y, aggfunc="size", fill_value=0)
                            im = ax.imshow(piv.values, aspect="auto", cmap=palette or "YlOrRd")
                            ax.set_xticks(range(len(piv.columns)))
                            ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
                            ax.set_yticks(range(len(piv.index)))
                            ax.set_yticklabels(piv.index, fontsize=7)
                            fig.colorbar(im, ax=ax, shrink=0.7)

                        elif chart_key == "boxplot":
                            if not y:
                                raise ValueError("Seleziona Asse Y (colonna numerica).")
                            if x:
                                cats = sorted(df_filt[x].dropna().unique(), key=str)
                                data = [pd.to_numeric(
                                    df_filt.loc[df_filt[x] == c, y], errors="coerce"
                                ).dropna() for c in cats]
                                bp = ax.boxplot(data, patch_artist=True)
                                for patch, col_ in zip(bp["boxes"], _colors(len(cats), palette)):
                                    patch.set_facecolor(col_)
                                ax.set_xticks(range(1, len(cats) + 1))
                                _smart_labels(ax, [str(c) for c in cats], xtick_rot)
                            else:
                                vals = pd.to_numeric(df_filt[y], errors="coerce").dropna()
                                ax.boxplot(vals, patch_artist=True)

                        _apply_style(ax, title, xlabel, ylabel, show_leg, show_grid)

                    fig.tight_layout(pad=1.5)
                    st.session_state.fig = fig

                except Exception as e:
                    st.error(f"Errore grafico: {e}")
                    st.session_state.fig = None

            # ── Mostra grafico + download ──────────────────────────────────────
            if st.session_state.fig is not None:
                st.pyplot(st.session_state.fig, use_container_width=True)

                buf = io.BytesIO()
                st.session_state.fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
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
