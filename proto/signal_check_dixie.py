"""Soft-goal signal check: CTrees AGB change vs OCR bp_2011, Dixie Fire window, H3 res 7."""
import numpy as np
import icechunk, xarray as xr
from h3ronpy.vector import coordinates_to_cells
from scipy import stats as sstats
import pyarrow as pa, pyarrow.parquet as pq
import time

T0 = time.time()
def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

LON0, LON1, LAT0, LAT1 = -122.0, -120.3, 39.6, 40.6
RES = 7

# ---- CTrees ----
log("opening CTrees")
st = icechunk.s3_storage(bucket="ctrees-agb-100m-global", prefix="agb_100m_global", region="us-west-2", anonymous=True)
sess = icechunk.Repository.open(st).readonly_session(branch="main")
ct = xr.open_zarr(sess.store, zarr_format=3, group="aboveground_biomass", chunks=None, decode_timedelta=False)
win = ct.agb.sel(y=slice(LAT1, LAT0), x=slice(LON0, LON1))
log(f"CTrees window shape (per year): {win.isel(time=0).shape}")

def yearmean(years):
    arrs = []
    for yr in years:
        a = win.sel(time=f"{yr}-01-01").values
        a = np.where(a <= -999, np.nan, a)   # nodata -9999
        arrs.append(a)
        log(f"  read {yr}")
    return np.nanmean(np.stack(arrs), axis=0) / 10.0  # int16 scale factor 0.1

early = yearmean([2001, 2002, 2003])
late  = yearmean([2023, 2024, 2025])
delta = late - early
log(f"delta computed; early mean {np.nanmean(early):.1f}, late mean {np.nanmean(late):.1f} Mg/ha")

yy = win.y.values; xx = win.x.values
gx, gy = np.meshgrid(xx, yy)
valid = ~np.isnan(delta)
cells_ct = np.asarray(coordinates_to_cells(gy[valid].ravel(), gx[valid].ravel(), RES))
dvals = delta[valid].ravel(); evals = early[valid].ravel()

def hexmean(cells, vals):
    uc, inv = np.unique(cells, return_inverse=True)
    sums = np.bincount(inv, weights=vals); cnts = np.bincount(inv)
    return uc, sums / cnts, cnts

uc_d, mean_delta, cnt_d = hexmean(cells_ct, dvals)
_, mean_early, _ = hexmean(cells_ct, evals)
log(f"CTrees hexes: {len(uc_d)}")

# ---- OCR ----
log("opening OCR")
st2 = icechunk.s3_storage(bucket="carbonplan", prefix="carbonplan-ocr/output/fire-risk/tensor/production/v1.1.0/ocr.icechunk",
                          endpoint_url="https://data.source.coop", region="us-west-2", anonymous=True, force_path_style=True)
sess2 = icechunk.Repository.open(st2).readonly_session(branch="main")
ocr = xr.open_zarr(sess2.store, zarr_format=3, chunks=None, decode_timedelta=False)
bp = ocr.bp_2011.sel(latitude=slice(LAT0, LAT1), longitude=slice(LON0, LON1))
if bp.sizes["latitude"] == 0:
    bp = ocr.bp_2011.sel(latitude=slice(LAT1, LAT0), longitude=slice(LON0, LON1))
bp = bp[::3, ::3]  # ~100m effective sampling is plenty for hex means
log(f"OCR window shape (subsampled): {bp.shape}")
bpv = bp.values
la = bp.latitude.values; lo = bp.longitude.values
glo, gla = np.meshgrid(lo, la)
ok = np.isfinite(bpv)
cells_bp = np.asarray(coordinates_to_cells(gla[ok].ravel(), glo[ok].ravel(), RES))
uc_b, mean_bp, cnt_b = hexmean(cells_bp, bpv[ok].ravel())
log(f"OCR hexes: {len(uc_b)}")

# ---- join ----
common, ia, ib = np.intersect1d(uc_d, uc_b, return_indices=True)
d = mean_delta[ia]; e = mean_early[ia]; b = mean_bp[ib]
log(f"joined hexes: {len(common)}")

def report(mask, label):
    dm, bm = d[mask], b[mask]
    if len(dm) < 30:
        print(f"{label}: only {len(dm)} hexes"); return
    rho, p = sstats.spearmanr(bm, dm)
    print(f"\n== {label} (n={len(dm)}) ==")
    print(f"Spearman rho(bp, delta) = {rho:.3f}  p = {p:.2e}")
    qs = np.quantile(bm, [0, .2, .4, .6, .8, 1.0])
    print(f"bp quintile edges: {np.array2string(qs, precision=4)}")
    for i in range(5):
        m2 = (bm >= qs[i]) & (bm <= qs[i+1] if i == 4 else bm < qs[i+1])
        print(f"  Q{i+1}: bp mean {bm[m2].mean():.4f}  delta mean {dm[m2].mean():+7.1f}  median {np.median(dm[m2]):+7.1f} Mg/ha  n={m2.sum()}")

report(np.ones_like(d, bool), "all hexes")
report(e >= 50, "forested hexes (early AGB >= 50 Mg/ha)")

# big-loss framing: do losing hexes sit at higher bp?
loss = d < -20
if loss.sum() > 30:
    u, pv = sstats.mannwhitneyu(b[loss], b[~loss], alternative="greater")
    print(f"\nbig-loss hexes (delta < -20): n={loss.sum()}, bp mean {b[loss].mean():.4f} vs others {b[~loss].mean():.4f}, Mann-Whitney p={pv:.2e}")

pq.write_table(pa.table({"h3": common, "delta_agb": d, "early_agb": e, "bp_2011": b}),
               "join/cache/hex_join_res7_dixie.parquet")
log("wrote join/cache/hex_join_res7_dixie.parquet")
