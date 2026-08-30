"""Hex tables for the waves view: burn probability (static, from bp_res7.parquet) and
FFWI per hour (from a prep_ffwi window) on one set of H3 cells, at res 7 (the label,
one HRRR pixel per cell, ring-filled where the cell holds no pixel centre) or res 6
(the parent, ~4 HRRR pixels and ~500 BP pixels per cell, mean and max both kept).
Nothing is resampled: every number is a group statistic over pixels keyed by label."""
import pathlib, time

import numpy as np, pyarrow.parquet as pq
from h3ronpy import change_resolution, grid_disk
from h3ronpy.vector import cells_to_wkb_polygons
import pyarrow as pa

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE, PROTO = ROOT / "join" / "cache", ROOT / "proto" / "cache"


def _group(keys):
    """sorted unique keys, inverse index (row -> group)."""
    u, inv = np.unique(keys, return_inverse=True)
    return u, inv.astype(np.int64)


def _reduce_frames(q, inv, K, how, step=16):
    """(F, N) uint8 FFWI in 0.5 steps (255 = nodata) grouped along N by inv into
    (F, K) uint8 of the same coding, `step` frames at a time."""
    o = np.argsort(inv, kind="stable"); g = inv[o]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]); cols = g[starts]
    F = q.shape[0]; out = np.full((F, K), 255, np.uint8)
    for f0 in range(0, F, step):
        qs = q[f0:f0 + step][:, o].astype(np.float32); qs[qs == 255] = np.nan
        if how == "max":
            r = np.fmax.reduceat(qs, starts, axis=1)
        else:
            n = np.add.reduceat(np.isfinite(qs).astype(np.float32), starts, axis=1)
            r = np.add.reduceat(np.nan_to_num(qs), starts, axis=1) / np.maximum(n, 1); r[n == 0] = np.nan
        out[f0:f0 + step, cols] = np.where(np.isnan(r), 255, np.rint(r)).astype(np.uint8)
    return out


def polygons(cells):
    """(positions float32 (V, 2) lon/lat, closed rings, startIndices uint32 (K+1)). Cells near an
    icosahedron edge carry distortion vertices (7 to 10 points), so the WKB is walked."""
    wkb = cells_to_wkb_polygons(pa.array(cells.astype(np.uint64)))
    raw = b"".join(wkb.to_pylist()); buf = np.frombuffer(raw, dtype=np.uint8)
    n = len(cells); starts = np.empty(n + 1, np.uint32); starts[0] = 0
    chunks = []; off = 0
    for i in range(n):
        npts = int.from_bytes(raw[off + 9:off + 13], "little")     # ring point count, closed
        chunks.append(buf[off + 13:off + 13 + npts * 16])          # closed ring, as deck wants it
        starts[i + 1] = starts[i] + npts
        off += 13 + npts * 16
    assert off == len(raw)
    xy = np.frombuffer(np.concatenate(chunks).tobytes(), dtype="<f8").reshape(-1, 2).astype(np.float32)
    return xy, starts


def build(res, window_npz, stats=True):
    t0 = time.time()
    bp = pq.read_table(CACHE / "bp_res7.parquet")
    c7 = bp["cell7"].to_numpy().astype(np.uint64); bmean = bp["bp_mean"].to_numpy(); bmax = bp["bp_max"].to_numpy(); bn = bp["n"].to_numpy().astype(np.float64)
    z = np.load(window_npz); q, lidx, times = z["ffwi"], z["lidx"], z["times"]
    label7 = np.load(PROTO / "label7.npy").ravel()[lidx]
    F, N = q.shape
    if res == 7:
        cells = c7
        # each BP cell takes its own HRRR pixel, else the mean / max over gridDisk(1)
        order = np.argsort(label7); lab_s = label7[order]
        disk = np.asarray(grid_disk(pa.array(cells), 1, flatten=True)).astype(np.uint64).reshape(len(cells), 7)
        pos = np.searchsorted(lab_s, disk); pos[pos >= lab_s.size] = 0
        hit = lab_s[pos] == disk                          # (K, 7) which disk cells hold a pixel
        pix = np.where(hit, order[pos], -1)              # land-pixel row or -1
        own = hit[:, 0]
        # rows for the mean over the disk: expand (cell, pixel) pairs, group by cell
        kk, dd = np.nonzero(hit)
        keep = own[kk] & (dd == 0) | ~own[kk]            # own pixel only if there is one, else the ring
        kk, pr = kk[keep], pix[kk[keep], dd[keep]]
        inv = kk.astype(np.int64)
        ffwi = {"mean": _reduce_frames(q[:, pr], inv, len(cells), "mean")}
        if stats: ffwi["max"] = _reduce_frames(q[:, pr], inv, len(cells), "max")
        bpv = {"mean": bmean.astype(np.float32), "max": bmax.astype(np.float32)}
        covered = ffwi["mean"][0] != 255
    elif res == 6:
        p6_bp = np.asarray(change_resolution(pa.array(c7), 6)).astype(np.uint64)
        p6_px = np.asarray(change_resolution(pa.array(label7), 6)).astype(np.uint64)
        cells = np.intersect1d(np.unique(p6_bp), np.unique(p6_px))
        # BP: n-weighted mean and max of the res 7 stats (exact: the res 7 mean is a sum / n)
        i_bp = np.searchsorted(cells, p6_bp); ok = (i_bp < cells.size); ok[ok] &= cells[i_bp[ok]] == p6_bp[ok]
        s = np.bincount(i_bp[ok], weights=(bmean * bn)[ok], minlength=cells.size); n = np.bincount(i_bp[ok], weights=bn[ok], minlength=cells.size)
        mx = np.full(cells.size, 0.0, np.float32); np.maximum.at(mx, i_bp[ok], bmax[ok])
        bpv = {"mean": (s / np.maximum(n, 1)).astype(np.float32), "max": mx}
        i_px = np.searchsorted(cells, p6_px); okp = (i_px < cells.size); okp[okp] &= cells[i_px[okp]] == p6_px[okp]
        ffwi = {"mean": _reduce_frames(q[:, okp], i_px[okp], cells.size, "mean")}
        if stats: ffwi["max"] = _reduce_frames(q[:, okp], i_px[okp], cells.size, "max")
        covered = np.ones(cells.size, bool)
    else:
        raise ValueError(res)
    cells = cells[covered]; bpv = {k: v[covered] for k, v in bpv.items()}; ffwi = {k: v[:, covered] for k, v in ffwi.items()}
    xy, starts = polygons(cells)
    return dict(res=res, cells=cells, bp=bpv, ffwi=ffwi, times=times, xy=xy, starts=starts, build_s=time.time() - t0)


if __name__ == "__main__":
    import sys
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    h = build(r, CACHE / "ffwi_2026-06-29_2026-07-05.npz")
    K = h["cells"].size
    print(f"res {r}: {K:,} cells, {h['ffwi']['mean'].shape[0]} frames, built in {h['build_s']:.1f}s; "
          f"ffwi mean bytes {h['ffwi']['mean'].nbytes/1e6:.0f} MB; xy {h['xy'].nbytes/1e6:.1f} MB")
    for k in ("mean", "max"):
        b = h["bp"][k]; f = h["ffwi"][k].astype(np.float32); f[f == 255] = np.nan
        print(f"  bp {k}: p50 {np.percentile(b,50):.5f} p90 {np.percentile(b,90):.4f} p99 {np.percentile(b,99):.4f} max {b.max():.4f}; "
              f"ffwi {k}: nan {np.isnan(f).mean():.4f} p50 {np.nanpercentile(f,50)/2:.1f} p99 {np.nanpercentile(f,99)/2:.1f}; verts {h['xy'].shape[0]:,}")
