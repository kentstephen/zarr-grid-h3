"""Fosberg Fire Weather Index per HRRR land pixel per hour for a window, from the
Dynamical HRRR analysis (2 m T and RH, 10 m wind), read block-wise through the disk
mirror. No resampling: one FFWI per pixel-hour, from that pixel's own T, RH, wind.

Fosberg (1978): m = equilibrium moisture content from RH (%) and T (F), piecewise;
eta = 1 - 2(m/30) + 1.5(m/30)^2 - 0.5(m/30)^3; FFWI = eta sqrt(1 + U^2) / 0.3002,
U wind in mph. Capped at 100.

Usage: uv run python join/prep_ffwi.py 2026-06-29 2026-07-05
Writes join/cache/ffwi_{d0}_{d1}.npz: times (F) datetime64[m], ffwi (F, N) uint8 in
0.5 steps (255 = no data), and lidx (N) uint32 = flat index of each land pixel in the
(1059, 1799) grid. Labels come from proto/cache/label7.npy at draw time.
"""
import pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hrrr_mirror import CHUNK_H, open_hrrr

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "join" / "cache"; PROTO = ROOT / "proto" / "cache"
VARS = ["temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m"]
B, THREADS = 45, 8
d0, d1 = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2026-06-29", "2026-07-05")

t00 = time.time()
def log(*a): print(f"[{time.time() - t00:6.1f}s]", *a, flush=True)

g, mirror = open_hrrr(VARS)
import xarray as xr
times = xr.open_zarr(mirror, consolidated=False, chunks=None)["time"].values.astype("datetime64[m]")
t0 = np.datetime64(d0).astype("datetime64[m]"); t1 = min((np.datetime64(d1) + np.timedelta64(23, "h")).astype("datetime64[m]"), times[-1])
i0, i1 = int(np.searchsorted(times, t0)), int(np.searchsorted(times, t1, side="right"))
F = i1 - i0
log(f"store {times[0]} to {times[-1]} ({times.size:,} h); window {times[i0]} to {times[i1 - 1]} = {F} h; time chunks {sorted({i // CHUNK_H for i in range(i0, i1)})}")

land = np.load(PROTO / "land.npy"); ny, nx = land.shape
lidx = np.flatnonzero(land.ravel()).astype(np.uint32); N = lidx.size
prow = np.full(ny * nx, -1, np.int64); prow[lidx] = np.arange(N); prow = prow.reshape(ny, nx)
blocks = []
for j in range(0, -(-ny // B)):
    for i in range(0, -(-nx // B)):
        ys, xs = slice(j * B, min((j + 1) * B, ny)), slice(i * B, min((i + 1) * B, nx))
        sub = prow[ys, xs].ravel(); loc = np.flatnonzero(sub >= 0)
        if loc.size: blocks.append((ys, xs, sub[loc], loc))
log(f"{N:,} land pixels in {len(blocks)} store blocks")

raw = {v: np.full((F, N), np.nan, np.float32) for v in VARS}
za = {v: g[v] for v in VARS}
def rd(job):
    v, (ys, xs, rows, loc) = job
    raw[v][:, rows] = za[v][i0:i1, ys, xs].reshape(F, -1)[:, loc]
with ThreadPoolExecutor(THREADS) as ex:
    list(ex.map(rd, [(v, b) for v in VARS for b in blocks]))
log(f"read {F * N * len(VARS):,} values; mirror hits {mirror.hits}, fetched {mirror.misses}")

T = raw["temperature_2m"] * 9.0 / 5.0 + 32.0
RH = np.clip(raw["relative_humidity_2m"], 0.0, 100.0)
U = np.hypot(raw["wind_u_10m"], raw["wind_v_10m"]) * 2.23694
m = np.where(RH < 10.0, 0.03229 + 0.281073 * RH - 0.000578 * RH * T,
    np.where(RH < 50.0, 2.22749 + 0.160107 * RH - 0.01478 * T,
             21.0606 + 0.005565 * RH * RH - 0.00035 * RH * T - 0.483199 * RH))
x = np.clip(m, 0.0, 30.0) / 30.0
eta = 1.0 - 2.0 * x + 1.5 * x * x - 0.5 * x * x * x
ffwi = np.clip(eta * np.sqrt(1.0 + U * U) / 0.3002, 0.0, 100.0)
q = np.where(np.isnan(ffwi), 255, np.rint(ffwi * 2.0)).astype(np.uint8)
log(f"ffwi: nan {np.isnan(ffwi).mean():.4f}; p50 {np.nanpercentile(ffwi, 50):.1f} p90 {np.nanpercentile(ffwi, 90):.1f} p99 {np.nanpercentile(ffwi, 99):.1f} max {np.nanmax(ffwi):.1f}; "
    f"hour max of pixel-max {np.nanmax(ffwi, axis=1).max():.1f}, share of pixel-hours >= 50: {(ffwi >= 50).mean():.4f}")
out = OUT / f"ffwi_{d0}_{d1}.npz"
np.savez(out, times=times[i0:i1], ffwi=q, lidx=lidx)
log("wrote", out, f"{out.stat().st_size / 1e6:.0f} MB")
