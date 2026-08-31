"""CONUS OCR fire-risk ingest: bp_2011, rps_scott, bp_2047 hex means at H3 res 7.
Tiled on the native (6000, 4500) chunk grid, subsampled ::3 (~90 m effective,
the pilot showed hex means at ~100 m sampling are fine). Checkpointed per tile.

Usage: uv run python conus/ingest_ocr.py [--test]
"""
import pathlib, sys, time

import numpy as np
import icechunk, xarray as xr
import pyarrow as pa, pyarrow.parquet as pq
from h3ronpy.vector import coordinates_to_cells

RES = 7
VARS = ["bp_2011", "rps_scott", "bp_2047"]
SUB = 3
CY, CX = 6000, 4500
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "cache" / "ocr_tiles"

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

def open_ocr():
    st = icechunk.s3_storage(
        bucket="carbonplan",
        prefix="carbonplan-ocr/output/fire-risk/tensor/production/v1.1.0/ocr.icechunk",
        endpoint_url="https://data.source.coop", region="us-west-2",
        anonymous=True, force_path_style=True)
    sess = icechunk.Repository.open(st).readonly_session(branch="main")
    return xr.open_zarr(sess.store, zarr_format=3, chunks=None, decode_timedelta=False)

def main(test=False):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = open_ocr()
    la = ds.latitude.values; lo = ds.longitude.values
    ny, nx = la.size, lo.size
    if test:
        iy = np.searchsorted(la, [39.6, 40.6]); ix = np.searchsorted(lo, [-122.0, -120.3])
        y0 = (iy[0] // CY) * CY; y1 = min(int(np.ceil(iy[1] / CY)) * CY, ny)
        x0 = (ix[0] // CX) * CX; x1 = min(int(np.ceil(ix[1] / CX)) * CX, nx)
    else:
        y0, y1, x0, x1 = 0, ny, 0, nx
    XGANG = 4                                  # chunks fetched per read, for concurrency
    tiles = [(y, x) for y in range(y0, y1, CY) for x in range(x0, x1, XGANG * CX)]
    log(f"rows {y0}..{y1} cols {x0}..{x1}: {len(tiles)} tile groups ({XGANG} chunks wide)")

    for ti, (ty, tx) in enumerate(tiles):
        name = f"tile_{ty}_{tx}.parquet"
        fp = OUT / name
        if fp.exists():
            log(f"{ti+1}/{len(tiles)} {name} exists, skip"); continue
        ye, xe = min(ty + CY, y1), min(tx + XGANG * CX, x1)
        sub = {}
        first = ds[VARS[0]].isel(latitude=slice(ty, ye), longitude=slice(tx, xe)).values[::SUB, ::SUB]
        if not np.isfinite(first).any():
            pq.write_table(pa.table({"h3": pa.array([], pa.uint64())}), fp)
            log(f"{ti+1}/{len(tiles)} {name} empty"); continue
        sub[VARS[0]] = first
        for v in VARS[1:]:
            sub[v] = ds[v].isel(latitude=slice(ty, ye), longitude=slice(tx, xe)).values[::SUB, ::SUB]
        lat_s = la[ty:ye:SUB]; lon_s = lo[tx:xe:SUB]
        anyok = np.zeros_like(first, dtype=bool)
        for v in VARS: anyok |= np.isfinite(sub[v])
        py, px = np.nonzero(anyok)
        cells = np.asarray(coordinates_to_cells(lat_s[py], lon_s[px].astype(np.float64), RES)).astype(np.uint64)
        uc, inv = np.unique(cells, return_inverse=True)
        cols = {"h3": uc}
        for v in VARS:
            vals = sub[v][anyok]; ok = np.isfinite(vals)
            cols[f"sum_{v}"] = np.bincount(inv, weights=np.where(ok, vals, 0).astype(np.float64), minlength=uc.size)
            cols[f"n_{v}"] = np.bincount(inv, weights=ok.astype(np.float64), minlength=uc.size).astype(np.int64)
        pq.write_table(pa.table(cols), fp)
        log(f"{ti+1}/{len(tiles)} {name}: {uc.size:,} hexes")
    log("done")

if __name__ == "__main__":
    main(test="--test" in sys.argv)
