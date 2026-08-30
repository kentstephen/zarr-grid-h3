"""Prepare the label layer + a 48 h HRRR slab for the raster-mesh prototype.

Writes to proto/cache/: corners_wm.npy (ny+1, nx+1, 2 float32 web-mercator world
coords, 512 units per world), label7.npy (ny, nx uint64), parent6.npy, land.npy (bool),
county_idx.npy (uint16 per pixel, 65535 none), county_names.json, hi_q.npy (F, ny, nx
uint8 heat index 0.5 degC steps from -40, 255 = no data), labels.json (frame stamps).
"""
import asyncio, hashlib, json, os, pathlib, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

import duckdb, icechunk, numpy as np, pyarrow as pa, pyarrow.parquet as pq, xarray as xr, zarr
from h3ronpy.vector import coordinates_to_cells
from h3ronpy import change_resolution
from pyproj import CRS, Transformer
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest

OUT = pathlib.Path(__file__).parent / "cache"; OUT.mkdir(exist_ok=True)
CACHE_DIR = tempfile.gettempdir() + "/x-sql-marimo"
MIRROR_DIR = CACHE_DIR + "/hrrr-mirror/v0.2.0.icechunk"
COUNTIES = CACHE_DIR + "/counties-2026-07-22.0-z8--124.8-24.4--66.9-49.5.parquet"
T0, T1 = "2026-07-01T00:00", "2026-07-02T23:00"   # East dome, in the mirrored chunk 47
RES_L, RES_T = 7, 6


class MirrorStore(Store):  # copied from x-sql-marimo/xsql-hrrr-heat-domes.py
    def __init__(self, inner, root, mirrorable):
        super().__init__(read_only=True); self.inner, self.root, self.mirrorable = inner, root, mirrorable
        self.hits = self.misses = 0; os.makedirs(root, exist_ok=True)
    supports_writes = supports_deletes = supports_partial_writes = False; supports_listing = True
    def __eq__(self, o): return isinstance(o, MirrorStore) and o.inner == self.inner and o.root == self.root
    def _tag(self, r):
        if r is None: return "all"
        if isinstance(r, RangeByteRequest): return f"r{r.start}-{r.end}"
        if isinstance(r, OffsetByteRequest): return f"o{r.offset}"
        if isinstance(r, SuffixByteRequest): return f"s{r.suffix}"
        return "x" + hashlib.sha1(repr(r).encode()).hexdigest()[:12]
    def _path(self, key, r): return os.path.join(self.root, key.replace("/", "__") + "." + self._tag(r))
    def _read(self, key, r, prototype):
        try:
            with open(self._path(key, r), "rb") as f: self.hits += 1; return prototype.buffer.from_bytes(f.read())
        except FileNotFoundError: return None
    def _write(self, key, r, buf):
        p = self._path(key, r); tmp = f"{p}.{os.getpid()}.{id(buf)}.tmp"
        with open(tmp, "wb") as f: f.write(buf.to_bytes())
        try: os.replace(tmp, p)
        except FileNotFoundError: pass
    async def get(self, key, prototype, byte_range=None):
        if not self.mirrorable(key): return await self.inner.get(key, prototype, byte_range)
        buf = await asyncio.to_thread(self._read, key, byte_range, prototype)
        if buf is not None: return buf
        self.misses += 1; buf = await self.inner.get(key, prototype, byte_range)
        if buf is not None: await asyncio.to_thread(self._write, key, byte_range, buf)
        return buf
    async def get_ranges(self, key, byte_ranges, *, prototype, max_concurrency=10, max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20):
        if not self.mirrorable(key):
            async for g in self.inner.get_ranges(key, byte_ranges, prototype=prototype, max_concurrency=max_concurrency, max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes): yield g
            return
        ranges = list(byte_ranges)
        held = await asyncio.gather(*(asyncio.to_thread(self._read, key, r, prototype) for r in ranges))
        hit = [(i, b) for i, b in enumerate(held) if b is not None]
        if hit: yield hit
        miss = [i for i, b in enumerate(held) if b is None]
        if not miss: return
        self.misses += len(miss)
        async for g in self.inner.get_ranges(key, [ranges[i] for i in miss], prototype=prototype, max_concurrency=max_concurrency, max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes):
            out = []
            for j, buf in g:
                i = miss[j]
                if buf is not None: await asyncio.to_thread(self._write, key, ranges[i], buf)
                out.append((i, buf))
            yield out
    async def get_partial_values(self, prototype, key_ranges): return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))
    async def exists(self, key): return await self.inner.exists(key)
    async def set(self, key, value): raise NotImplementedError
    async def delete(self, key): raise NotImplementedError
    def list(self): return self.inner.list()
    def list_prefix(self, p): return self.inner.list_prefix(p)
    def list_dir(self, p): return self.inner.list_dir(p)
    async def getsize(self, key): return await self.inner.getsize(key)


t = time.perf_counter()
storage = icechunk.s3_storage(bucket="dynamical-noaa-hrrr", prefix="noaa-hrrr-analysis/v0.2.0.icechunk", region="us-west-2", anonymous=True)
sess = icechunk.Repository.open(storage).readonly_session("main")
VARS = ["temperature_2m", "relative_humidity_2m"]
T = zarr.open_group(sess.store, mode="r")["time"].shape[0]; young = (T - 1) // 2160
def mirrorable(key, _v=set(VARS)):
    p = key.split("/"); return len(p) == 5 and p[0] in _v and p[1] == "c" and p[2].isdigit() and int(p[2]) < young
store = MirrorStore(sess.store, MIRROR_DIR, mirrorable)
ds = xr.open_zarr(store, consolidated=False, chunks=None)
lat = ds["latitude"].values.astype("float64"); lon = ds["longitude"].values.astype("float64")
gx = ds["x"].values; gy = ds["y"].values; ny, nx = lat.shape
print(f"store open {time.perf_counter()-t:.1f}s grid {ny}x{nx}")

# --- the mesh: pixel corners in the native LCC grid -> lon/lat -> web mercator world coords
t = time.perf_counter()
crs = CRS.from_wkt(ds["spatial_ref"].attrs["crs_wkt"])
dx = float(gx[1] - gx[0]); dy = float(gy[1] - gy[0])
cx = np.concatenate([gx - dx / 2, [gx[-1] + dx / 2]]); cy = np.concatenate([gy - dy / 2, [gy[-1] + dy / 2]])
CX, CY = np.meshgrid(cx, cy)
to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
clon, clat = to_ll.transform(CX.ravel(), CY.ravel())
# web mercator world coords, 512 units per world, y down (deck's CARTESIAN world space is y up: use +)
wx = (clon + 180.0) / 360.0 * 512.0
wy = (1.0 - np.log(np.tan(np.radians(clat)) + 1.0 / np.cos(np.radians(clat))) / np.pi) / 2.0 * 512.0
corners = np.stack([wx, 512.0 - wy], axis=-1).astype(np.float32).reshape(ny + 1, nx + 1, 2)
np.save(OUT / "corners_wm.npy", corners)
print(f"corners {time.perf_counter()-t:.1f}s")

# --- the label layer: one res 7 cell per pixel (from the store's own lat/lon), its res 6 parent
t = time.perf_counter()
label7 = np.asarray(coordinates_to_cells(lat.ravel(), lon.ravel(), RES_L)).astype(np.uint64).reshape(ny, nx)
parent6 = np.asarray(change_resolution(pa.array(label7.ravel()), RES_T)).astype(np.uint64).reshape(ny, nx)
print(f"labels {time.perf_counter()-t:.1f}s unique res7 {np.unique(label7).size:,} res6 {np.unique(parent6).size:,}")

# --- land mask + county per pixel: counties polyfilled at res 6 (center), pixel is land if its parent is in one
t = time.perf_counter()
con = duckdb.connect(); con.sql("INSTALL h3 FROM community; LOAD h3; INSTALL spatial; LOAD spatial;")
counties = pq.read_table(COUNTIES); con.register("c", counties)
m = con.sql("""
  WITH parts AS (SELECT id, UNNEST(ST_Dump(ST_GeomFromWKB(wkb))).geom AS g FROM c),
  filled AS (SELECT id, UNNEST(h3_polygon_wkb_to_cells_experimental(ST_AsWKB(g), 6, 'center')) AS hex FROM parts)
  SELECT hex, any_value(id) AS id FROM filled GROUP BY hex ORDER BY hex""").to_arrow_table()
cells6 = m["hex"].to_numpy().astype(np.uint64)
pos = np.searchsorted(cells6, parent6.ravel()); pos[pos >= cells6.size] = 0
land = (cells6[pos] == parent6.ravel()).reshape(ny, nx)
cid = counties["id"].to_pylist(); cpos = {i: k for k, i in enumerate(cid)}
cell_county = np.fromiter((cpos.get(i, 65535) for i in m["id"].to_pylist()), dtype=np.uint16, count=cells6.size)
county_idx = np.full(ny * nx, 65535, np.uint16); county_idx[land.ravel()] = cell_county[pos[land.ravel()]]
names = [f"{n}, {r}" for n, r in zip(counties["name"].to_pylist(), counties["region"].to_pylist())]
np.save(OUT / "label7.npy", label7); np.save(OUT / "parent6.npy", parent6); np.save(OUT / "land.npy", land)
np.save(OUT / "county_idx.npy", county_idx.reshape(ny, nx)); (OUT / "county_names.json").write_text(json.dumps(names))
print(f"land {time.perf_counter()-t:.1f}s {land.sum():,} land px, {cells6.size:,} land cells")

# --- the slab: only the 45x45 store blocks that touch land, from the mirror, all threads
t = time.perf_counter()
times = ds["time"].values.astype("datetime64[m]")
i0 = int(np.searchsorted(times, np.datetime64(T0))); i1 = int(np.searchsorted(times, np.datetime64(T1))) + 1
B = 45; by, bx = -(-ny // B), -(-nx // B)
blocks = [(j, i) for j in range(by) for i in range(bx) if land[j*B:(j+1)*B, i*B:(i+1)*B].any()]
za = {v: zarr.open_group(store, mode="r")[v] for v in VARS}
out = {v: np.full((i1 - i0, ny, nx), np.nan, np.float32) for v in VARS}
def rd(args):
    v, j, i = args
    out[v][:, j*B:(j+1)*B, i*B:(i+1)*B] = za[v][i0:i1, j*B:(j+1)*B, i*B:(i+1)*B]
with ThreadPoolExecutor(8) as ex: list(ex.map(rd, [(v, j, i) for v in VARS for j, i in blocks]))
print(f"slab {time.perf_counter()-t:.1f}s {i1-i0} h x {len(blocks)} blocks; mirror hits {store.hits} misses {store.misses}")

def heat_index_c(tc, rh):
    T = tc * 9.0 / 5.0 + 32.0; hi = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + rh * 0.094)
    mm = (hi + T) / 2.0 >= 80.0; T2, R2 = T[mm], rh[mm]
    h = (-42.379 + 2.04901523*T2 + 10.14333127*R2 - 0.22475541*T2*R2 - 0.00683783*T2*T2 - 0.05481717*R2*R2
         + 0.00122874*T2*T2*R2 + 0.00085282*T2*R2*R2 - 0.00000199*T2*T2*R2*R2)
    a1 = (R2 < 13) & (T2 >= 80) & (T2 <= 112); h[a1] -= ((13 - R2[a1]) / 4.0) * np.sqrt((17 - np.abs(T2[a1] - 95.0)) / 17.0)
    a2 = (R2 > 85) & (T2 >= 80) & (T2 <= 87); h[a2] += ((R2[a2] - 85.0) / 10.0) * ((87.0 - T2[a2]) / 5.0)
    hi[mm] = h; return (hi - 32.0) * 5.0 / 9.0
hi = heat_index_c(out["temperature_2m"].astype(np.float64), out["relative_humidity_2m"].astype(np.float64)).astype(np.float32)
hi[:, ~land] = np.nan
ok = np.isfinite(hi); hi_q = np.full(hi.shape, 255, np.uint8); hi_q[ok] = np.clip(np.rint((hi[ok] + 40) * 2), 0, 254)
np.save(OUT / "hi_q.npy", hi_q)
(OUT / "labels.json").write_text(json.dumps([np.datetime_as_string(x, unit="m").replace("T", " ") + "Z" for x in times[i0:i1]]))
print(f"heat index {np.nanmin(hi):.1f}..{np.nanmax(hi):.1f} degC, median {np.nanmedian(hi):.1f}; hi_q {hi_q.nbytes/1e6:.0f} MB")
