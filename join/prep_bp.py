"""USFS burn probability (Dillon et al. 2023, FSim, 270 m NAD83 Conus Albers) rolled up
to H3 res 7: the many-to-one case of the label join. Each 270 m pixel is labelled from
its projected centre; ~70 pixels fall in a res 7 cell (5.16 km2 / 0.0729 km2), and the
cell gets their mean, max and count. Nothing is resampled; the label is the group key.

Source (source.coop, icechunk):
  carbonplan/carbonplan-ocr/input/fire-risk/tensor/USFS/dillon-et-al-2023/
  processed-270m-5070.icechunk  -> BP (11283, 17372) float32, chunks (6000, 4500),
  fill -3.4e38, 0 = non-burnable. Zeros stay in the mean (a cell that is half rock
  burns half as often); fill is dropped.

Writes join/cache/bp_res7.parquet: cell7 uint64, bp_mean, bp_max float32, n uint16.
"""
import base64, pathlib, struct, time

import icechunk, numpy as np, pyarrow as pa, pyarrow.parquet as pq, zarr
from h3ronpy.vector import coordinates_to_cells
from pyproj import Transformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "join" / "cache"; OUT.mkdir(exist_ok=True)
RES = 7
SC = "us-west-2.opendata.source.coop"
PREFIX = "carbonplan/carbonplan-ocr/input/fire-risk/tensor/USFS/dillon-et-al-2023/processed-270m-5070.icechunk"
BAND = 1500  # rows per in-memory pass inside a chunk

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)

st = icechunk.s3_storage(bucket=SC, prefix=PREFIX, region="us-west-2", anonymous=True)
g = zarr.open_group(icechunk.Repository.open(st).readonly_session("main").store, mode="r")
bp, x, y = g["BP"], g["x"][:], g["y"][:]
fill = struct.unpack("<d", base64.b64decode(bp.attrs["_FillValue"]))[0]
log("BP", bp.shape, "chunks", bp.chunks, "fill", fill, "dx", x[1] - x[0], "dy", y[1] - y[0])
to_ll = Transformer.from_crs(g["spatial_ref"].attrs["crs_wkt"], "EPSG:4326", always_xy=True)

def reduce(cells, vals, n=None):
    """Group (cells, vals[, n]) by cell -> sorted unique cells, sum, count, max."""
    o = np.argsort(cells, kind="stable")
    c, v = cells[o], vals[o]
    cnt = np.ones_like(v, dtype=np.int64) if n is None else n[o]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return c[starts], np.add.reduceat(v, starts), np.add.reduceat(cnt, starts), np.maximum.reduceat(v, starts)

parts = []  # (cell, sum, cnt, max) per band
cy, cx = bp.chunks
for r0 in range(0, bp.shape[0], cy):
    for c0 in range(0, bp.shape[1], cx):
        blk = bp[r0:r0 + cy, c0:c0 + cx]
        log(f"chunk rows {r0}:{r0 + blk.shape[0]} cols {c0}:{c0 + blk.shape[1]}")
        for b0 in range(0, blk.shape[0], BAND):
            v = blk[b0:b0 + BAND]
            ok = v != fill
            if not ok.any(): continue
            ii, jj = np.nonzero(ok)
            lon, lat = to_ll.transform(x[c0 + jj], y[r0 + b0 + ii])
            cells = np.asarray(coordinates_to_cells(lat, lon, RES)).astype(np.uint64)
            parts.append(reduce(cells, v[ok].astype(np.float64)))

log("merging", len(parts), "bands")
cell = np.concatenate([p[0] for p in parts]); s = np.concatenate([p[1] for p in parts])
n = np.concatenate([p[2] for p in parts]); mx = np.concatenate([p[3] for p in parts])
# second pass: a cell straddling a band or chunk edge appears more than once
o = np.argsort(cell, kind="stable"); cell, s, n, mx = cell[o], s[o], n[o], mx[o]
starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
cell = cell[starts]; s = np.add.reduceat(s, starts); n = np.add.reduceat(n, starts); mx = np.maximum.reduceat(mx, starts)
mean = s / n
log(f"{len(cell):,} cells; n per cell min {n.min()} median {np.median(n):.0f} max {n.max()}")
log(f"bp_mean: >0 share {np.mean(mean > 0):.3f}, p50 {np.percentile(mean, 50):.5f}, p90 {np.percentile(mean, 90):.5f}, p99 {np.percentile(mean, 99):.5f}, max {mean.max():.4f}")
log(f"bp_max:  p50 {np.percentile(mx, 50):.5f}, p99 {np.percentile(mx, 99):.5f}, max {mx.max():.4f}")
pq.write_table(pa.table({"cell7": cell, "bp_mean": mean.astype(np.float32), "bp_max": mx.astype(np.float32), "n": n.astype(np.uint16)}), OUT / "bp_res7.parquet")
log("wrote", OUT / "bp_res7.parquet")
