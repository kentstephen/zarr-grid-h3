"""Decomposition and stats: does bp separate losses among hexes that actually
burned, and is the unburned-unharvested residual noise or signal?

Classes on the forested subset (early AGB >= 50 Mg/ha), thresholded on the
fraction of the hex's 343 res-10 cells touched:
  burned    frac_wf >= 0.10  (wildfire incl. Wildland Fire Use, ignition 2001+)
  harvested frac_harvest >= 0.10  (FACTS completed harvest/thin, federal only)
  both / neither. "neither" is not "no cause": private harvest is unlabeled.

Output: conus/cache/hex_conus_res7_labeled.parquet + printed report.
Usage: uv run python conus/decompose.py
"""
import pathlib

import numpy as np
import pyarrow.parquet as pq, pyarrow as pa
from scipy import stats as sstats
from h3ronpy.vector import cells_to_coordinates

ROOT = pathlib.Path(__file__).resolve().parent
FRAC_T = 0.10
N10 = 343.0

def partial_spearman(x, y, z):
    """rank-based partial corr of x, y controlling z"""
    rx, ry, rz = (sstats.rankdata(v) for v in (x, y, z))
    def resid(a, b):
        A = np.c_[b, np.ones_like(b)]
        return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    return np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1]

def quintiles(b, d, label):
    qs = np.quantile(b, [0, .2, .4, .6, .8, 1.0])
    rows = []
    for i in range(5):
        m = (b >= qs[i]) & ((b < qs[i+1]) if i < 4 else (b <= qs[i+1]))
        rows.append(f"    Q{i+1}: bp {b[m].mean():.4f}  delta median {np.median(d[m]):+7.1f}  mean {d[m].mean():+7.1f}  n={m.sum():,}")
    print(f"  bp_2011 quintiles ({label}):"); print("\n".join(rows))

def report(m, tag, b, d, e, r, b47):
    n = m.sum()
    if n < 200:
        print(f"== {tag}: n={n} (too few) =="); return
    rho, p = sstats.spearmanr(b[m], d[m])
    rho_r, _ = sstats.spearmanr(r[m], d[m])
    rho47, _ = sstats.spearmanr(b47[m], d[m])
    pr = partial_spearman(b[m], d[m], e[m])
    print(f"== {tag} (n={n:,}) ==")
    print(f"  rho(bp_2011, delta)={rho:+.3f} p={p:.1e}   partial|early={pr:+.3f}   rps_scott {rho_r:+.3f}   bp_2047 {rho47:+.3f}")
    quintiles(b[m], d[m], tag)

def main():
    t = pq.read_table(ROOT / "cache" / "hex_conus_res7.parquet")
    lab = pq.read_table(ROOT / "cache" / "labels_res7.parquet")
    h = t["h3"].to_numpy(); hl = lab["h3"].to_numpy().astype(np.uint64)
    idx = np.searchsorted(hl, h)          # labels written sorted? enforce
    o = np.argsort(hl); hl = hl[o]
    idx = np.searchsorted(hl, h); idx[idx >= hl.size] = 0
    hit = hl[idx] == h
    def col(name, fill=0):
        v = lab[name].to_numpy(zero_copy_only=False).astype(np.float64)[o]
        out = np.full(h.size, np.nan if fill is None else float(fill))
        out[hit] = v[idx[hit]]
        return out
    frac_wf = col("n10_wf") / N10
    frac_rx = col("n10_rx") / N10
    frac_hv = col("n10_harvest") / N10
    last_ig = col("last_ig_wf", fill=None)

    d = t["delta_agb"].to_numpy(); e = t["early_agb"].to_numpy()
    b = t["bp_2011"].to_numpy(); r = t["rps_scott"].to_numpy(); b47 = t["bp_2047"].to_numpy()
    burned = frac_wf >= FRAC_T; harv = frac_hv >= FRAC_T
    cls = np.where(burned & harv, 3, np.where(burned, 2, np.where(harv, 1, 0))).astype(np.int8)
    forest = e >= 50

    ll = cells_to_coordinates(pa.array(h.astype(np.uint64)))
    lon = np.asarray(ll["lng"] if "lng" in ll.column_names else ll["lon"])
    lat = np.asarray(ll["lat"])

    print(f"hexes: {h.size:,}  forested: {forest.sum():,}")
    print(f"class counts (forested): neither {np.sum(forest & (cls==0)):,}  harvested {np.sum(forest & (cls==1)):,}  "
          f"burned {np.sum(forest & (cls==2)):,}  both {np.sum(forest & (cls==3)):,}")
    print(f"burned frac_wf>= {FRAC_T}; rx-touched forested hexes: {np.sum(forest & (frac_rx >= FRAC_T)):,} (kept in their class; sensitivity below)\n")

    report(forest, "forested, all", b, d, e, r, b47)
    for c, tag in [(0, "neither (unburned, unharvested)"), (1, "harvested only"), (2, "burned only"), (3, "burned + harvested")]:
        report(forest & (cls == c), tag, b, d, e, r, b47)
    report(forest & (cls == 0) & (frac_rx < FRAC_T), "neither, excl. prescribed-burn hexes", b, d, e, r, b47)

    print("\n-- regions (forested) --")
    for tag, m0 in [("Pacific (lon<-115)", lon < -115), ("Mountain (-115..-104)", (lon >= -115) & (lon < -104)),
                    ("Central (-104..-90)", (lon >= -104) & (lon < -90)), ("East (lon>=-90)", lon >= -90)]:
        for c, ctag in [(2, "burned"), (0, "neither")]:
            m = forest & m0 & (cls == c)
            if m.sum() >= 200:
                rho, _ = sstats.spearmanr(b[m], d[m])
                print(f"  {tag:24s} {ctag:8s} n={m.sum():7,}  rho={rho:+.3f}  delta median {np.median(d[m]):+6.1f}")

    print("\n-- did bp_2047 move risk toward the fires? (forested) --")
    for ctag, m in [("burned", forest & burned), ("unburned", forest & ~burned)]:
        dd = b47[m] - b[m]
        print(f"  {ctag:8s} n={m.sum():7,}  median bp_2011 {np.median(b[m]):.4f}  median bp_2047-bp_2011 {np.median(dd):+.4f}  ratio median {np.median(b47[m]/np.maximum(b[m],1e-6)):.2f}")
    au = sstats.mannwhitneyu(b47[forest & burned] - b[forest & burned],
                             b47[forest & ~burned] - b[forest & ~burned], alternative="greater")
    print(f"  Mann-Whitney (burned delta-bp > unburned): p={au.pvalue:.1e}")

    out = ROOT / "cache" / "hex_conus_res7_labeled.parquet"
    cols = {c: t[c].to_numpy() for c in t.column_names}
    cols.update(frac_wf=frac_wf, frac_rx=frac_rx, frac_harvest=frac_hv,
                last_ig_wf=last_ig, cls=cls, lon=lon, lat=lat)
    pq.write_table(pa.table(cols), out)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
