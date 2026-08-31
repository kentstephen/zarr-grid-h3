"""Probe: dynamical's MRMS CONUS analysis hourly (source.coop, plain zarr v3, anonymous).

Plan doc 11 step 1: variables/units, grid origin and chunking, time range and lag,
a box read (rate, NaN story over ocean). Also a glance at the HRRR forecast-48-hour
store's shape, for the future forecast fence. Numbers go back into docs/11.
"""
import time

import numpy as np
import zarr
from obstore.store import HTTPStore
from zarr.storage import ObjectStore

SC = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/dynamical"
MRMS_URL = f"{SC}/noaa-mrms-conus-analysis-hourly/v0.3.0.zarr"
HRRR_F48_URL = f"{SC}/noaa-hrrr-forecast-48-hour/v0.1.0.zarr"

T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def open_group(url):
    return zarr.open_group(ObjectStore(HTTPStore.from_url(url), read_only=True), mode="r")


# ---- MRMS -------------------------------------------------------------------------
log("opening MRMS", MRMS_URL)
g = open_group(MRMS_URL)
for name, arr in sorted(g.arrays()):
    a = dict(arr.attrs)
    log(f"  {name}: {arr.shape} {arr.dtype} chunks={arr.chunks} shards={getattr(arr, 'shards', None)}"
        f" fill={arr.fill_value!r} units={a.get('units')!r} long_name={a.get('long_name')!r}")
log("group attrs:", dict(g.attrs))

t = g["time"]
tattrs = dict(t.attrs)
log("time attrs:", tattrs)
tv = t[:]
log(f"time: n={tv.shape[0]} first={tv[0]} last={tv[-1]} step={tv[1]-tv[0]}")
# decode against epoch attrs if numeric
units = tattrs.get("units", "")
if np.issubdtype(np.asarray(tv).dtype, np.number) and "since" in units:
    import datetime as dt
    base = dt.datetime.fromisoformat(units.split("since")[1].strip().replace("Z", "+00:00"))
    stepu = units.split()[0]
    mult = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}[stepu]
    first = base + dt.timedelta(seconds=float(tv[0]) * mult)
    last = base + dt.timedelta(seconds=float(tv[-1]) * mult)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) if base.tzinfo is None else dt.datetime.now(dt.timezone.utc)
    log(f"time decoded: {first} .. {last} (lag from now: {now - last})")

lat = g["latitude"][:]; lon = g["longitude"][:]
log(f"lat: n={lat.shape[0]} {lat[0]:.4f}..{lat[-1]:.4f} step={lat[1]-lat[0]:.5f}")
log(f"lon: n={lon.shape[0]} {lon[0]:.4f}..{lon[-1]:.4f} step={lon[1]-lon[0]:.5f}")

# ---- box read: last 24 h over a Gulf-coast storm box (land + ocean in frame) ------
data_vars = [n for n, arr in g.arrays() if arr.ndim == 3]
log("3d vars:", data_vars)
v = g["precipitation_surface"]  # NOT data_vars[0]: that is the categorical type var (codes, -3 sentinel)
BOX = (-98.0, 26.0, -88.0, 33.0)  # TX/LA gulf coast: land, coast, open water
i0 = int(np.argmin(np.abs(lat - BOX[3]))); i1 = int(np.argmin(np.abs(lat - BOX[1])))
if i0 > i1: i0, i1 = i1, i0
j0 = int(np.argmin(np.abs(lon - BOX[0]))); j1 = int(np.argmin(np.abs(lon - BOX[2])))
T = v.shape[0]
log(f"box rows {i0}..{i1} ({i1-i0}) cols {j0}..{j1} ({j1-j0}), last 24 h of {T}")
t_read = time.time()
block = v[T-24:T, i0:i1, j0:j1]
dtb = time.time() - t_read
px = block.size
log(f"read {block.shape} = {px:,} px in {dtb:.1f}s ({px/dtb/1e6:.1f} Mpx/s), dtype {block.dtype}")
bf = block.astype(np.float64)
nan = np.isnan(bf)
log(f"NaN fraction: {nan.mean():.4f}")
bf *= 3600.0  # kg m-2 s-1 -> mm/h
wet = bf > 0.1
log(f"finite: min={np.nanmin(bf):.3f} max={np.nanmax(bf):.3f} mean={np.nanmean(bf):.4f}; "
    f"wet(>0.1) fraction of finite: {wet.sum()/max(1,(~nan).sum()):.4f}")
# ocean corner: SE of the box is open gulf
oc = bf[:, -80:, -80:]
log(f"open-gulf corner NaN fraction: {np.isnan(oc).mean():.4f}, max={np.nanmax(oc) if not np.all(np.isnan(oc)) else float('nan'):.3f}")

# ---- HRRR forecast-48-hour: shape only --------------------------------------------
log("opening HRRR forecast-48h", HRRR_F48_URL)
try:
    gf = open_group(HRRR_F48_URL)
    for name, arr in sorted(gf.arrays()):
        if "precip" in name or arr.ndim >= 3 or name in ("time", "lead_time", "init_time"):
            log(f"  {name}: {arr.shape} {arr.dtype} chunks={arr.chunks}")
except Exception as e:
    log("forecast store failed:", e)

log("done")
