"""Step 6 viz, univariate panels only (no bivariate encodings), protan-safe:
no red anywhere, meaning never rides on red-vs-green.

  shots/conus_delta_map.png   delta AGB, diverging blue (gain) / orange (loss)
  shots/conus_bp_map.png      bp_2011, cividis
  shots/conus_class_map.png   cause classes, 4 colorblind-safe hues + legend
  shots/conus_quintiles.png   bp quintile vs median delta, per class

Usage: uv run python conus/viz.py
"""
import pathlib

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = pathlib.Path(__file__).resolve().parent
SHOTS = ROOT / "shots"

# blue (gain) <- white -> orange (loss): both legs survive protan simulation
BLOR = LinearSegmentedColormap.from_list("blor", ["#b35806", "#f5f5f5", "#2166ac"])
CLS_COLORS = {0: "#9aa0a6", 1: "#7b3294", 2: "#e08214", 3: "#0571b0"}
CLS_NAMES = {0: "neither", 1: "harvested", 2: "burned", 3: "both"}

def basemap(ax):
    ax.set_facecolor("#ffffff")
    ax.set_aspect(1.25)          # rough CONUS lat/lon aspect
    ax.set_xlim(-125.5, -66.5); ax.set_ylim(24, 50)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def main():
    t = pq.read_table(ROOT / "cache" / "hex_conus_res7_labeled.parquet")
    g = {c: t[c].to_numpy() for c in t.column_names}
    forest = g["early_agb"] >= 50
    lon, lat = g["lon"], g["lat"]
    d, b, cls = g["delta_agb"], g["bp_2011"], g["cls"]
    print(f"{forest.sum():,} forested hexes drawn")

    fig, ax = plt.subplots(figsize=(13, 7), dpi=200); basemap(ax)
    lim = np.nanpercentile(np.abs(d[forest]), 98)
    sc = ax.scatter(lon[forest], lat[forest], c=d[forest], s=0.4, cmap=BLOR,
                    norm=TwoSlopeNorm(0, -lim, lim), linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="delta AGB 2001-03 to 2023-25 (Mg/ha)")
    ax.set_title("CTrees biomass change, forested res-7 hexes (early AGB >= 50 Mg/ha)")
    fig.savefig(SHOTS / "conus_delta_map.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=200); basemap(ax)
    sc = ax.scatter(lon[forest], lat[forest], c=b[forest], s=0.4, cmap="cividis",
                    vmin=0, vmax=np.nanpercentile(b[forest], 98), linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="bp_2011 (annual burn probability)")
    ax.set_title("CarbonPlan OCR modeled burn probability, same hexes")
    fig.savefig(SHOTS / "conus_bp_map.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=200); basemap(ax)
    for c in (0, 1, 2, 3):
        m = forest & (cls == c)
        ax.scatter(lon[m], lat[m], s=0.4 if c == 0 else 0.8, c=CLS_COLORS[c],
                   linewidths=0, rasterized=True, label=f"{CLS_NAMES[c]} ({m.sum():,})")
    ax.legend(loc="lower left", markerscale=20, frameon=False)
    ax.set_title("Cause labels: MTBS wildfire burn vs FACTS federal harvest (frac >= 0.1 of hex)")
    fig.savefig(SHOTS / "conus_class_map.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    for c in (0, 1, 2, 3):
        m = forest & (cls == c)
        if m.sum() < 200: continue
        bm, dm = b[m], d[m]
        qs = np.quantile(bm, [0, .2, .4, .6, .8, 1.0])
        xs, ys = [], []
        for i in range(5):
            mm = (bm >= qs[i]) & ((bm < qs[i+1]) if i < 4 else (bm <= qs[i+1]))
            xs.append(bm[mm].mean()); ys.append(np.median(dm[mm]))
        ax.plot(xs, ys, marker=markers[c], color=CLS_COLORS[c], label=f"{CLS_NAMES[c]} (n={m.sum():,})")
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("bp_2011 quintile mean (log scale)"); ax.set_ylabel("median delta AGB (Mg/ha)")
    ax.set_title("Does modeled burn probability separate observed loss?\nforested hexes, by cause class")
    ax.legend(frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(SHOTS / "conus_quintiles.png", bbox_inches="tight"); plt.close(fig)
    print(f"wrote 4 panels to {SHOTS}")

if __name__ == "__main__":
    main()
