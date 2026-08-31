"""Merge the per-tile hex sums into one res-7 table and inner-join CTrees x OCR.
Output: conus/cache/hex_conus_res7.parquet
Columns: h3, delta_agb, early_agb, late_agb, mean_unc, n_px, bp_2011, rps_scott, bp_2047.

Usage: uv run python conus/reduce_join.py [--validate]
--validate additionally reruns the pilot stats (forested Spearman, quintiles)
over whatever tiles exist, for comparison with proto/signal_check_dixie.py.
"""
import pathlib, sys

import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import pyarrow.dataset as pads

ROOT = pathlib.Path(__file__).resolve().parent

def merged(tiledir, sumcols):
    ds = pads.dataset(sorted(tiledir.glob("tile_*.parquet")))
    t = ds.to_table()
    if t.num_rows == 0:
        raise SystemExit(f"no rows under {tiledir}")
    h = t["h3"].to_numpy()
    uc, inv = np.unique(h, return_inverse=True)
    out = {"h3": uc}
    for c in sumcols:
        out[c] = np.bincount(inv, weights=t[c].to_numpy().astype(np.float64), minlength=uc.size)
    return out

def main(validate=False):
    ct = merged(ROOT / "cache" / "ctrees_tiles", ["sum_early", "sum_late", "sum_unc", "n_unc", "n"])
    n = ct["n"]
    early = ct["sum_early"] / n
    late = ct["sum_late"] / n
    unc = np.divide(ct["sum_unc"], ct["n_unc"], out=np.full_like(early, np.nan), where=ct["n_unc"] > 0)
    print(f"CTrees hexes: {len(n):,}")

    oc = merged(ROOT / "cache" / "ocr_tiles", ["sum_bp_2011", "n_bp_2011", "sum_rps_scott", "n_rps_scott", "sum_bp_2047", "n_bp_2047"])
    print(f"OCR hexes: {len(oc['h3']):,}")

    common, ia, ib = np.intersect1d(ct["h3"], oc["h3"], return_indices=True)
    cols = {"h3": common,
            "delta_agb": (late - early)[ia], "early_agb": early[ia], "late_agb": late[ia],
            "mean_unc": unc[ia], "n_px": n[ia].astype(np.int64)}
    for v in ["bp_2011", "rps_scott", "bp_2047"]:
        m = np.divide(oc[f"sum_{v}"], oc[f"n_{v}"], out=np.full(len(oc["h3"]), np.nan), where=oc[f"n_{v}"] > 0)
        cols[v] = m[ib]
    keep = np.isfinite(cols["bp_2011"])
    cols = {k: v[keep] for k, v in cols.items()}
    print(f"joined hexes (finite bp_2011): {len(cols['h3']):,}")
    out = ROOT / "cache" / "hex_conus_res7.parquet"
    pq.write_table(pa.table(cols), out)
    print(f"wrote {out}")

    if validate:
        from scipy import stats as sstats
        d, e, b = cols["delta_agb"], cols["early_agb"], cols["bp_2011"]
        for mask, label in [(np.ones_like(d, bool), "all"), (e >= 50, "forested (early >= 50)")]:
            dm, bm = d[mask], b[mask]
            rho, p = sstats.spearmanr(bm, dm)
            print(f"{label}: n={mask.sum():,} rho(bp_2011, delta)={rho:.3f} p={p:.1e}")
            qs = np.quantile(bm, [0, .2, .4, .6, .8, 1.0])
            for i in range(5):
                m2 = (bm >= qs[i]) & ((bm < qs[i+1]) if i < 4 else (bm <= qs[i+1]))
                print(f"  Q{i+1}: bp {bm[m2].mean():.4f}  delta median {np.median(dm[m2]):+7.1f}  n={m2.sum()}")

if __name__ == "__main__":
    main(validate="--validate" in sys.argv)
