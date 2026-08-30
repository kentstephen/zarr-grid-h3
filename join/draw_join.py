"""Draw the HRRR x CONUS404 join at pixel scale for one small window, four panels.
Reads join/cache; writes shots/join_*.png. Colours are luminance-only (protan-safe)."""
import pathlib, numpy as np, pyarrow.parquet as pq, h3
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pyproj import CRS, Transformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
C = ROOT / "join" / "cache"; OUT = ROOT / "shots"; OUT.mkdir(exist_ok=True)
C404 = CRS.from_proj4("+proj=lcc +lat_1=30 +lat_2=50 +lat_0=39.1 +lon_0=-97.9 +R=6370000 +units=m +no_defs")
HRRR = CRS.from_proj4("+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m +no_defs")
h2c = Transformer.from_crs(HRRR, C404, always_xy=True)
ll2c = Transformer.from_crs("EPSG:4326", C404, always_xy=True)

hlat = np.load(C / "hrrr_lat.npy"); hlon = np.load(C / "hrrr_lon.npy")
p99_on = np.load(C / "p99_on_hrrr.npy"); c99 = np.load(C / "c404_p99.npy")
T = pq.read_table(C / "join.parquet")
class Row:
    def __init__(self, k): self.k = k
    def __getattr__(self, n): return T.column(n)[self.k].as_py()
JIDX = {(a, b): k for k, (a, b) in enumerate(zip(T.column("hi").to_numpy(), T.column("hj").to_numpy()))}
HX0, HY0, HD = -2697520.14, 1586693.85, 3000.0
cx = np.arange(c99.shape[1]) * 4000.0 + 0; cy = np.arange(c99.shape[0]) * 4000.0
# recover CONUS404 x/y origin from the labels: use lat/lon of pixel (0,0) is not cached; derive from prep: x = -2732000 + 4000 j? check with join rows instead
# CONUS404 centres: reconstruct from a matched pixel pair (dist known) is fragile; instead read x/y offsets from the p99 store shape + known WRF origin
CX0, CY0 = -2732000.0, -2028000.0   # WRF CONUS404 (x[0], y[0]) in metres; verified below against join distances

# ---- window: 12x12 HRRR pixels around a point in Kansas ---------------------------
lat0, lon0 = 38.55, -98.6
d = (hlat - lat0) ** 2 + (hlon - lon0) ** 2
i0, j0 = np.unravel_index(np.argmin(d), d.shape)
N = 6
ii = np.arange(i0 - N, i0 + N); jj = np.arange(j0 - N, j0 + N)

def hrrr_square(i, j):
    x = HX0 + j * HD; y = HY0 - i * HD
    xs = [x - HD/2, x + HD/2, x + HD/2, x - HD/2]; ys = [y - HD/2, y - HD/2, y + HD/2, y + HD/2]
    X, Y = h2c.transform(xs, ys); return np.c_[X, Y]
def hrrr_centre(i, j):
    return ll2c.transform(hlon[i, j], hlat[i, j])
def c404_square(ci, cj):
    x = CX0 + cj * 4000; y = CY0 + ci * 4000
    return np.array([[x-2000, y-2000], [x+2000, y-2000], [x+2000, y+2000], [x-2000, y+2000]])
def c404_centre(ci, cj): return CX0 + cj * 4000, CY0 + ci * 4000
def hexagon(cell):
    b = h3.cell_to_boundary(cell)  # (lat, lng)
    X, Y = ll2c.transform([p[1] for p in b], [p[0] for p in b]); return np.c_[X, Y]

rows = [(i, j, Row(JIDX[(i, j)])) for i in ii for j in jj if (i, j) in JIDX]
# sanity: our reconstructed CONUS404 centre should match dist_m in the table
i, j, r = rows[0]
hx, hy = hrrr_centre(i, j); cxm, cym = c404_centre(int(r.ci), int(r.cj))
print(f"origin check: table dist {r.dist_m:.0f} m, reconstructed {np.hypot(hx-cxm, hy-cym):.0f} m")

cset = {(int(r.ci), int(r.cj)) for _, _, r in rows}
cells = {int(r.cell7) for _, _, r in rows}
cells_hex = {h3.int_to_str(c) for c in cells}
hx_all = np.array([hrrr_centre(i, j) for i, j, _ in rows])
xlim = (hx_all[:, 0].min() - 3000, hx_all[:, 0].max() + 3000); ylim = (hx_all[:, 1].min() - 3000, hx_all[:, 1].max() + 3000)

def base(ax, title):
    ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
def draw_grids(ax, hrrr_fill=None, c404_fill=None):
    for ci, cj in cset:
        fc = "none" if c404_fill is None else c404_fill(ci, cj)
        ax.add_patch(Polygon(c404_square(ci, cj), fc=fc, ec="#1f4e79", lw=2.0, zorder=2))
    for i, j, r in rows:
        fc = "none" if hrrr_fill is None else hrrr_fill(i, j, r)
        ax.add_patch(Polygon(hrrr_square(i, j), fc=fc, ec="#c98a00", lw=0.9, zorder=3))
def draw_centres(ax):
    for ci, cj in cset: ax.plot(*c404_centre(ci, cj), "s", ms=6, color="#1f4e79", zorder=6)
    for i, j, _ in rows: ax.plot(*hrrr_centre(i, j), "o", ms=3.5, color="#c98a00", zorder=6)
def legend(ax, extra=()):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#c98a00", lw=1, marker="o", ms=4, label="HRRR pixel, 3 km (centre = dot)"),
         Line2D([], [], color="#1f4e79", lw=2, marker="s", ms=5, label="CONUS404 pixel, 4 km (centre = square)")] + list(extra)
    ax.legend(handles=h, loc="upper left", fontsize=8, framealpha=0.95)

from matplotlib.lines import Line2D
# 1. the two grids, nothing else
fig, ax = plt.subplots(figsize=(9, 9)); base(ax, "1. Two grids over the same ground, Kansas. Neither lines up with the other.")
draw_grids(ax); draw_centres(ax); legend(ax); fig.savefig(OUT / "join_1_grids.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# 2. add the res 7 hexagons
fig, ax = plt.subplots(figsize=(9, 9)); base(ax, "2. H3 res 7 hexagons: each HRRR centre gets the hexagon it sits in (its label).")
draw_grids(ax)
for c in cells_hex: ax.add_patch(Polygon(hexagon(c), fc="none", ec="#555555", lw=0.8, ls="-", zorder=4))
draw_centres(ax); legend(ax, [Line2D([], [], color="#555555", lw=0.8, label="res 7 hexagon (~1.3 km across)")])
fig.savefig(OUT / "join_2_hexagons.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# 3. the match: direct hexagons shaded, arrows to partner
fig, ax = plt.subplots(figsize=(9, 9)); base(ax, "3. The join: each HRRR centre is linked to one CONUS404 centre.\n   Shaded hexagon = both centres in the same hexagon (direct). Unshaded = partner found in a neighbouring hexagon (ring).")
draw_grids(ax)
for i, j, r in rows:
    c = h3.int_to_str(int(r.cell7))
    ax.add_patch(Polygon(hexagon(c), fc="#bbbbbb" if r.how == 0 else "none", ec="#555555", lw=0.8, alpha=0.6, zorder=4))
for i, j, r in rows:
    hx, hy = hrrr_centre(i, j); cxm, cym = c404_centre(int(r.ci), int(r.cj))
    ax.annotate("", xy=(cxm, cym), xytext=(hx, hy), arrowprops=dict(arrowstyle="->", lw=1.1 if r.how == 0 else 0.8, color="#111111" if r.how == 0 else "#888888"), zorder=5)
draw_centres(ax)
legend(ax, [Line2D([], [], color="#111111", lw=1.1, label="direct link (same hexagon)"), Line2D([], [], color="#888888", lw=0.8, label="ring link (neighbour hexagon)")])
fig.savefig(OUT / "join_3_links.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# 4. what B looks like: the CONUS404 value copied onto HRRR pixels, side by side
vals = np.array([c99[int(r.ci), int(r.cj)] for _, _, r in rows]); vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
cmap = plt.get_cmap("viridis"); norm = plt.Normalize(vmin, vmax)
fig, axs = plt.subplots(1, 2, figsize=(16, 8.5))
base(axs[0], "4a. View A: CONUS404 p99 Fosberg on CONUS404 pixels (as published)")
for ci, cj in cset: axs[0].add_patch(Polygon(c404_square(ci, cj), fc=cmap(norm(c99[ci, cj])), ec="#1f4e79", lw=1.2, zorder=2))
base(axs[1], "4b. View B: the same numbers on HRRR pixels, each copied from its partner")
for i, j, r in rows: axs[1].add_patch(Polygon(hrrr_square(i, j), fc=cmap(norm(p99_on[i, j])), ec="#c98a00", lw=0.8, zorder=3))
for ax in axs:
    for ci, cj in cset: ax.add_patch(Polygon(c404_square(ci, cj), fc="none", ec="#1f4e79", lw=1.2, zorder=4))
    ax.plot([xlim[0]+2000, xlim[0]+2000+20000], [ylim[0]+2000]*2, color="#111", lw=3); ax.text(xlim[0]+2000, ylim[0]+3000, "20 km", fontsize=9)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.6, label="CONUS404 p99 FFWI")
fig.savefig(OUT / "join_4_values.png", dpi=110, bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("join_*.png")))
