"""The Dynamical HRRR analysis store behind the disk mirror shared with the notebooks
(copy of MirrorStore from hrrr-heat-domes.py; same mirror dir, same file names)."""
import asyncio, hashlib, os, tempfile

import icechunk, zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest

BUCKET = "dynamical-noaa-hrrr"
PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
MIRROR_DIR = tempfile.gettempdir() + "/x-sql-marimo/hrrr-mirror/" + PREFIX.split("/")[-1]
CHUNK_H = 2160


class MirrorStore(Store):
    def __init__(self, inner, root, mirrorable):
        super().__init__(read_only=True)
        self.inner, self.root, self.mirrorable = inner, root, mirrorable
        self.hits = self.misses = 0
        os.makedirs(root, exist_ok=True)

    supports_writes = supports_deletes = supports_partial_writes = False
    supports_listing = True

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
            with open(self._path(key, r), "rb") as f:
                self.hits += 1
                return prototype.buffer.from_bytes(f.read())
        except FileNotFoundError:
            return None

    def _write(self, key, r, buf):
        p = self._path(key, r); tmp = f"{p}.{os.getpid()}.{id(buf)}.tmp"
        with open(tmp, "wb") as f: f.write(buf.to_bytes())
        try: os.replace(tmp, p)
        except FileNotFoundError: pass

    async def get(self, key, prototype, byte_range=None):
        if not self.mirrorable(key): return await self.inner.get(key, prototype, byte_range)
        buf = await asyncio.to_thread(self._read, key, byte_range, prototype)
        if buf is not None: return buf
        self.misses += 1
        buf = await self.inner.get(key, prototype, byte_range)
        if buf is not None: await asyncio.to_thread(self._write, key, byte_range, buf)
        return buf

    async def get_ranges(self, key, byte_ranges, *, prototype, max_concurrency=10, max_gap_bytes=1 << 20, max_coalesced_bytes=16 << 20):
        kw = dict(prototype=prototype, max_concurrency=max_concurrency, max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes)
        if not self.mirrorable(key):
            async for group in self.inner.get_ranges(key, byte_ranges, **kw): yield group
            return
        ranges = list(byte_ranges)
        held = await asyncio.gather(*(asyncio.to_thread(self._read, key, r, prototype) for r in ranges))
        hit = [(i, b) for i, b in enumerate(held) if b is not None]
        if hit: yield hit
        miss = [i for i, b in enumerate(held) if b is None]
        if not miss: return
        self.misses += len(miss)
        async for group in self.inner.get_ranges(key, [ranges[i] for i in miss], **kw):
            out = []
            for j, buf in group:
                if buf is not None: await asyncio.to_thread(self._write, key, ranges[miss[j]], buf)
                out.append((miss[j], buf))
            yield out

    async def get_partial_values(self, prototype, key_ranges):
        return list(await asyncio.gather(*(self.get(k, prototype, r) for k, r in key_ranges)))

    async def exists(self, key): return await self.inner.exists(key)
    async def set(self, key, value): raise NotImplementedError
    async def delete(self, key): raise NotImplementedError
    def list(self): return self.inner.list()
    def list_prefix(self, p): return self.inner.list_prefix(p)
    def list_dir(self, p): return self.inner.list_dir(p)
    async def getsize(self, key): return await self.inner.getsize(key)


def open_hrrr(variables, cache_gb=2):
    """(zarr group, MirrorStore) for the given variables; full time shards of closed
    chunks are mirrored to disk, the youngest chunk is always fetched."""
    st = icechunk.s3_storage(bucket=BUCKET, prefix=PREFIX, region="us-west-2", anonymous=True)
    sess = icechunk.Repository.open(
        st, config=icechunk.RepositoryConfig(caching=icechunk.CachingConfig(num_bytes_chunks=int(cache_gb * (1 << 30))))
    ).readonly_session("main")
    T = zarr.open_group(sess.store, mode="r")["time"].shape[0]
    young, mv = (T - 1) // CHUNK_H, set(variables)

    def mirrorable(key):
        p = key.split("/")
        return len(p) == 5 and p[0] in mv and p[1] == "c" and p[2].isdigit() and int(p[2]) < young

    mirror = MirrorStore(sess.store, MIRROR_DIR, mirrorable)
    return zarr.open_group(mirror, mode="r"), mirror
