"""CONUS CTrees ingest: early (2001-03) and late (2023-25) AGB means plus late
uncertainty, tiled on the native (2000, 2000) chunk grid, hex-aggregated to H3
res 7 as sums and counts per tile. Checkpointed: a tile parquet that exists is
skipped, so the run resumes after interruption. Merge happens in reduce_join.py.

Usage: uv run python conus/ingest_ctrees.py [--test]
--test restricts to the pilot Dixie window for validation against
proto/signal_check_dixie.py numbers.
"""
import pathlib, sys, time

import numpy as np
import icechunk, xarray as xr
import pyarrow as pa, pyarrow.parquet as pq
from h3ronpy.vector import coordinates_to_cells

RES = 7
EARLY, LATE = [2001, 2002, 2003], [2023, 2024, 2025]
# OCR extent, snapped outward to the CTrees (2000, 2000) chunk grid.
# Global grid: y descending from 90, x ascending from -180, 1125 px/deg.
LAT0, LAT1, LON0, LON1 = 22.43, 52.48, -128.4, -64.05
CHUNK, TILE = 2000, 4          # tile = 4x4 chunks = 8000 x 8000 px
ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "cache" / "ctrees_tiles"

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

def open_agb():
    st = icechunk.s3_storage(bucket="ctrees-agb-100m-global", prefix="agb_100m_global",
                             region="us-west-2", anonymous=True)
    sess = icechunk.Repository.open(st).readonly_session(branch="main")
    return xr.open_zarr(sess.store, zarr_format=3, group="aboveground_biomass",
                        chunks=None, decode_timedelta=False)

def snap(v, up):
    return int(np.ceil(v / CHUNK) * CHUNK) if up else int(np.floor(v / CHUNK) * CHUNK)

def main(test=False, shard=0, nshards=1):
    OUT.mkdir(parents=True, exist_ok=True)
    ds = open_agb()
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    if test:
        la0, la1, lo0, lo1 = 39.6, 40.6, -122.0, -120.3
    else:
        la0, la1, lo0, lo1 = LAT0, LAT1, LON0, LON1
    iy0 = snap((90 - la1) * 1125, up=False); iy1 = min(snap((90 - la0) * 1125, up=True), ny)
    ix0 = snap((lo0 + 180) * 1125, up=False); ix1 = min(snap((lo1 + 180) * 1125, up=True), nx)
    step = CHUNK * TILE
    tiles = [(y, x) for y in range(iy0, iy1, step) for x in range(ix0, ix1, step)]
    tiles = tiles[shard::nshards]
    log(f"window rows {iy0}..{iy1} cols {ix0}..{ix1}: {len(tiles)} tiles of {step}x{step} (shard {shard}/{nshards})")

    for ti, (ty, tx) in enumerate(tiles):
        name = f"tile_{ty}_{tx}.parquet"
        fp = OUT / name
        if fp.exists():
            log(f"{ti+1}/{len(tiles)} {name} exists, skip"); continue
        ye, xe = min(ty + step, iy1), min(tx + step, ix1)
        win = ds.agb.isel(y=slice(ty, ye), x=slice(tx, xe))
        uwin = ds.uncertainty.isel(y=slice(ty, ye), x=slice(tx, xe))

        def accum(var, years):
            s = None; c = None
            for yr in years:
                a = var.sel(time=f"{yr}-01-01").values.astype(np.float32)
                a[a <= -999] = np.nan                     # nodata -9999
                v = np.isfinite(a)
                if s is None:
                    s = np.where(v, a, 0.0); c = v.astype(np.int16)
                else:
                    s += np.where(v, a, 0.0); c += v
                if yr == years[0] and not v.any():
                    return None, None                     # ocean tile: bail early
            m = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
            return m / 10.0, c                            # scale 0.1 -> Mg/ha

        early, _ = accum(win, EARLY)
        if early is None:
            pq.write_table(pa.table({"h3": pa.array([], pa.uint64())}), fp)
            log(f"{ti+1}/{len(tiles)} {name} empty (ocean)"); continue
        late, _ = accum(win, LATE)
        unc, _ = accum(uwin, LATE)

        valid = np.isfinite(early) & np.isfinite(late)
        if not valid.any():
            pq.write_table(pa.table({"h3": pa.array([], pa.uint64())}), fp)
            log(f"{ti+1}/{len(tiles)} {name} empty (no joint coverage)"); continue
        iy, ix = np.nonzero(valid)
        cells = np.asarray(coordinates_to_cells(
            win.y.values[iy].astype(np.float64), win.x.values[ix].astype(np.float64), RES)).astype(np.uint64)
        e = early[valid]; l = late[valid]
        u = unc[valid]; uok = np.isfinite(u)

        uc, inv = np.unique(cells, return_inverse=True)
        n = np.bincount(inv)
        tbl = pa.table({
            "h3": uc,
            "sum_early": np.bincount(inv, weights=e.astype(np.float64)),
            "sum_late": np.bincount(inv, weights=l.astype(np.float64)),
            "sum_unc": np.bincount(inv, weights=np.where(uok, u, 0).astype(np.float64)),
            "n_unc": np.bincount(inv, weights=uok.astype(np.float64)).astype(np.int64),
            "n": n.astype(np.int64),
        })
        pq.write_table(tbl, fp)
        log(f"{ti+1}/{len(tiles)} {name}: {len(uc):,} hexes from {valid.sum():,} px")
    log("done")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sh, ns = (int(args[0]), int(args[1])) if len(args) == 2 else (0, 1)
    main(test="--test" in sys.argv, shard=sh, nshards=ns)
