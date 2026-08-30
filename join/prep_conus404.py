"""HRRR x CONUS404: two Lambert grids matched pixel to pixel through their H3 res 7 labels.

Both grids are unique at res 7 (HRRR 3 km = 9 km2, CONUS404 4 km = 16 km2, res 7 cell
5.16 km2), so a res 7 cell holds at most one pixel of each. The join is an equi-join on
the label; cells where the coarser grid has no pixel are filled from gridDisk(1).
Nothing is resampled. Every match is then checked against the true nearest CONUS404
pixel (KD-tree in CONUS404 metres) so the cost of "nearest in the hierarchy" versus
"nearest in metres" is measured, not assumed.

Sources (all source.coop):
  HRRR lat/lon:   dynamical/noaa-hrrr-analysis/v0.2.0.zarr (zarr v3 mirror), latitude/longitude
  CONUS404 FFWI:  carbonplan/carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi/
                  fosberg-fire-weather-index-{p99,p95}.icechunk (one chunk each)
The CONUS404 stores carry an empty `crs`; the WRF CONUS404 Lambert is applied and
verified against the store's own 2D lat/lon corners.

Writes join/cache/:
  c404_label7.npy (1015, 1367) uint64, c404_p99.npy, c404_p95.npy (float32),
  hrrr_lat.npy / hrrr_lon.npy (1059, 1799) float32,
  join.parquet: one row per HRRR land pixel: hi, hj, cell7, ci, cj, how (0 direct,
  1 ring, 2 none), dist_m (matched centres), nn_ci, nn_cj, nn_dist_m (true nearest),
  p99_on_hrrr.npy (1059, 1799) float32 (NaN where no match).
"""
import pathlib, sys, time

import icechunk, numpy as np, pyarrow as pa, pyarrow.parquet as pq, zarr
from h3ronpy import grid_disk
from h3ronpy.vector import coordinates_to_cells
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "join" / "cache"; OUT.mkdir(exist_ok=True)
PROTO = ROOT / "proto" / "cache"
RES = 7
SC = "us-west-2.opendata.source.coop"
HRRR_URL = f"https://s3.us-west-2.amazonaws.com/{SC}/dynamical/noaa-hrrr-analysis/v0.2.0.zarr"
C404_PREFIX = "carbonplan/carbonplan-ocr/input/fire-risk/tensor/conus404-ffwi"
# WRF CONUS404 projection (Rasmussen et al. 2023): Lambert conformal, sphere R = 6370 km
C404_CRS = CRS.from_proj4("+proj=lcc +lat_1=30 +lat_2=50 +lat_0=39.1 +lon_0=-97.9 +R=6370000 +units=m +no_defs")

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def c404_group(name):
    st = icechunk.s3_storage(bucket=SC, prefix=f"{C404_PREFIX}/{name}.icechunk", region="us-west-2", anonymous=True)
    return zarr.open_group(icechunk.Repository.open(st).readonly_session("main").store, mode="r")


# ---- CONUS404: grid, projection check, labels, percentiles ------------------------
g99 = c404_group("fosberg-fire-weather-index-p99")
cx = g99["x"][:]; cy = g99["y"][:]                       # 4000 m, y ascending (row 0 south)
p99 = g99["FFWI"][:].astype(np.float32)
p95 = c404_group("fosberg-fire-weather-index-p95")["FFWI"][:].astype(np.float32)
ny4, nx4 = p99.shape
log(f"CONUS404 {ny4}x{nx4}, x {cx[0]:.0f}..{cx[-1]:.0f} dx {cx[1]-cx[0]:.0f}, y {cy[0]:.0f}..{cy[-1]:.0f} dy {cy[1]-cy[0]:.0f}")

to_ll = Transformer.from_crs(C404_CRS, "EPSG:4326", always_xy=True)
XX, YY = np.meshgrid(cx, cy)
clon, clat = to_ll.transform(XX.ravel(), YY.ravel())
clon = clon.reshape(ny4, nx4); clat = clat.reshape(ny4, nx4)

# verify against the store's own lat/lon at the corners and a few interior chunks (10x10 chunks)
checks = [(0, 0), (0, nx4 - 1), (ny4 - 1, 0), (ny4 - 1, nx4 - 1), (ny4 // 2, nx4 // 2), (300, 900), (700, 200)]
worst = 0.0
for (i, j) in checks:
    la = float(g99["lat"][i, j]); lo = float(g99["lon"][i, j])
    d = np.hypot((la - clat[i, j]) * 111_000, (lo - clon[i, j]) * 111_000 * np.cos(np.radians(la)))
    worst = max(worst, d)
    log(f"  crs check ({i},{j}) store ({la:.5f},{lo:.5f}) proj ({clat[i,j]:.5f},{clon[i,j]:.5f}) off {d:.1f} m")
if worst > 50:
    sys.exit(f"CONUS404 projection guess off by {worst:.0f} m; stop")
log(f"CONUS404 projection verified (worst {worst:.1f} m)")

c404_label = np.asarray(coordinates_to_cells(clat.ravel(), clon.ravel(), RES)).astype(np.uint64)
u4 = np.unique(c404_label).size
log(f"CONUS404 res {RES} labels: {c404_label.size:,} pixels, {u4:,} unique cells ({c404_label.size - u4} collisions)")
np.save(OUT / "c404_label7.npy", c404_label.reshape(ny4, nx4))
np.save(OUT / "c404_p99.npy", p99); np.save(OUT / "c404_p95.npy", p95)

# ---- HRRR: lat/lon from the source.coop mirror, labels from proto/cache ---------------
from obstore.store import HTTPStore
from zarr.storage import ObjectStore
hz = zarr.open_group(ObjectStore(HTTPStore.from_url(HRRR_URL), read_only=True), mode="r")
hlat = hz["latitude"][:].astype(np.float64); hlon = hz["longitude"][:].astype(np.float64)
np.save(OUT / "hrrr_lat.npy", hlat.astype(np.float32)); np.save(OUT / "hrrr_lon.npy", hlon.astype(np.float32))
ny3, nx3 = hlat.shape
hrrr_label = np.load(PROTO / "label7.npy")
land = np.load(PROTO / "land.npy")
assert hrrr_label.shape == (ny3, nx3)
# the cached labels came from the icechunk copy; confirm the mirror's lat/lon give the same labels
chk = np.asarray(coordinates_to_cells(hlat[::37].ravel(), hlon[::37].ravel(), RES)).astype(np.uint64)
assert np.array_equal(chk, hrrr_label[::37].ravel()), "mirror lat/lon disagree with cached labels"
log(f"HRRR {ny3}x{nx3}, {land.sum():,} land pixels; mirror lat/lon reproduce the cached res {RES} labels")

# ---- the join --------------------------------------------------------------------
hi, hj = np.nonzero(land)
hcell = hrrr_label[hi, hj]
# CONUS404 lookup: cell -> flat pixel index (unique, so a plain sorted search)
order = np.argsort(c404_label); c_sorted = c404_label[order]
def lookup(cells):
    pos = np.searchsorted(c_sorted, cells)
    pos = np.minimum(pos, c_sorted.size - 1)
    hit = c_sorted[pos] == cells
    return np.where(hit, order[pos], -1)

direct = lookup(hcell)
how = np.where(direct >= 0, 0, 2).astype(np.int8)
match = direct.copy()
log(f"direct (same res {RES} cell): {(direct >= 0).sum():,} of {hcell.size:,} HRRR land pixels")

# ring fill for the rest: gridDisk(1) of the HRRR label, keep the ring cell whose CONUS404
# pixel centre is nearest in metres (the hierarchy offers up to 6 candidates; metres pick one)
to_c404 = Transformer.from_crs("EPSG:4326", C404_CRS, always_xy=True)
hx, hy = to_c404.transform(hlon[hi, hj], hlat[hi, hj])           # HRRR centres in CONUS404 metres
miss = np.nonzero(direct < 0)[0]
if miss.size:
    disks = pa.array(grid_disk(pa.array(hcell[miss]), 1))          # ListArray, one disk per input
    offs = np.asarray(disks.offsets); flat = np.asarray(disks.values).astype(np.uint64)
    assert offs.size == miss.size + 1
    K = int(np.diff(offs).max())                                    # 7 (6 at a pentagon)
    ring = np.zeros((miss.size, K), np.uint64); okr = np.zeros((miss.size, K), bool)
    for r in range(K):
        n = np.diff(offs); has = n > r
        ring[has, r] = flat[offs[:-1][has] + r]; okr[has, r] = True
    cand = lookup(ring.ravel()).reshape(miss.size, K)
    ok = (cand >= 0) & okr
    ci_c = np.where(ok, cand // nx4, 0); cj_c = np.where(ok, cand % nx4, 0)
    d = np.hypot(cx[cj_c] - hx[miss, None], cy[ci_c] - hy[miss, None])
    d[~ok] = np.inf
    best = d.argmin(axis=1)
    got = np.isfinite(d[np.arange(miss.size), best])
    match[miss[got]] = cand[np.arange(miss.size), best][got]
    how[miss[got]] = 1
log(f"ring fill (gridDisk 1): {(how == 1).sum():,}; no partner: {(how == 2).sum():,}")

ci = np.where(match >= 0, match // nx4, -1); cj = np.where(match >= 0, match % nx4, -1)
dist = np.where(match >= 0, np.hypot(cx[np.maximum(cj, 0)] - hx, cy[np.maximum(ci, 0)] - hy), np.nan)

# truth: nearest CONUS404 pixel in metres
tree = cKDTree(np.c_[XX.ravel(), YY.ravel()])
nn_d, nn_idx = tree.query(np.c_[hx, hy], workers=-1)
agree = match == nn_idx
log(f"H3 match == nearest in metres: {agree.sum():,} / {match.size:,} ({100 * agree.mean():.2f}%)")
for name, sel in (("direct", how == 0), ("ring", how == 1)):
    if sel.any():
        log(f"  {name}: agree {100 * agree[sel].mean():.2f}%, centre distance median {np.nanmedian(dist[sel]):.0f} m, "
            f"p95 {np.nanpercentile(dist[sel], 95):.0f} m, max {np.nanmax(dist[sel]):.0f} m; nearest median {np.median(nn_d[sel]):.0f} m")
extra = dist - nn_d
log(f"extra distance paid by the hierarchy: median {np.nanmedian(extra):.0f} m, p95 {np.nanpercentile(extra, 95):.0f} m, max {np.nanmax(extra):.0f} m")

# reverse: CONUS404 pixels whose cell holds an HRRR pixel (any HRRR pixel, land or not)
rev = np.isin(c404_label, hrrr_label.ravel())
log(f"CONUS404 pixels with an HRRR pixel in their cell: {rev.sum():,} / {c404_label.size:,}")

# ---- outputs ------------------------------------------------------------------------
p99_on = np.full((ny3, nx3), np.nan, np.float32)
p99_on[hi[match >= 0], hj[match >= 0]] = p99.ravel()[match[match >= 0]]
np.save(OUT / "p99_on_hrrr.npy", p99_on)
tbl = pa.table({
    "hi": hi.astype(np.int32), "hj": hj.astype(np.int32), "cell7": hcell,
    "ci": ci.astype(np.int32), "cj": cj.astype(np.int32), "how": how, "dist_m": dist.astype(np.float32),
    "nn_ci": (nn_idx // nx4).astype(np.int32), "nn_cj": (nn_idx % nx4).astype(np.int32), "nn_dist_m": nn_d.astype(np.float32),
})
pq.write_table(tbl, OUT / "join.parquet", compression="zstd")
log(f"wrote {OUT}/join.parquet ({tbl.num_rows:,} rows) and p99_on_hrrr.npy")
