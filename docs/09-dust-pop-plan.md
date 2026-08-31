# 09: SILAM dust x Kontur population, worldwide

The idea: worldwide dust with population underneath. Join Kontur population cells
(cell_to_parent) to the dust grid on H3. Two readings of the same table: extrusion
= population with dust as color (who is standing under it), and flat color-only
(the plume itself). Variable: `cnc_dust`.

This is a refresh of a proven workflow, not a new one. The prior notebooks are:

- `~/dev/projects/duckdb-zarr-test/xarray-sql-duckdb.ipynb`: SILAM dust, April 2025
  Mongolia->China event, from the STALE `silam_global_dust_v3.zarr`, xarray-sql ->
  DuckDB (community h3 ext) -> Kontur r6 rollup -> kepler.gl parquet.
- `~/dev/projects/duckdb-zarr-test/firesmoke-duckdb.ipynb`: same chassis on the
  BlueSky smoke zarr. Both write (hex, ts, value, pop) parquet that kepler reads
  directly (H3 layer on `hex`, time filter on `ts`).

## The store (verified 2026-08-30)

`s3://bkr/silam-dust/silam_global_dust.icechunk` via https://data.source.coop,
anonymous, path-style. This is the LIVE store: new snapshot most mornings
06:00-09:00 UTC (commits Aug 25, 26, 28, 29, 30), ~947 snapshots. The three plain
zarr stores next to it are frozen (v3.zarr last touched 2025-10-01); the old
notebook's URL is dead weight. `silam_aerosol.icechunk` is dead since 2026-01-07.

- Grid: regular lonlat 0.1 deg exactly, 3600 x 1778, lat -88.85..88.85.
- Dims: init_time (465 daily 00Z runs, 2024-11-14 -> 2026-08-30, HAS GAPS but the
  last 6 days are contiguous) x step (1..120 hourly) x lat x lon.
- 23 vars; ours is `cnc_dust` (kg/m3, ~1e-7 scale; no units attr on this store).
- Chunking: (1, 1, 1778, 3600), one global field per init+step, ~13 MB. So N
  frames = N sequential chunk reads; ~2 chunks/s against source.coop was the
  measured rate before (it 500s under concurrent reads, keep target_partitions=1).

Open in the notebook:

```python
storage = icechunk.s3_storage(bucket="bkr",
    prefix="silam-dust/silam_global_dust.icechunk",
    endpoint_url="https://data.source.coop", region="us-west-2",
    anonymous=True, force_path_style=True)
repo = icechunk.Repository.open(storage)
ds = xr.open_zarr(repo.readonly_session("main").store,
                  consolidated=False, decode_timedelta=False)
```

## Stack (Stephen's call)

xarray-sql (DataFusion) with an h3ronpy UDF for the dust side; DuckDB with the
community h3 extension for the Kontur rollup and the join. Already done: `uv add
xarray-sql` in this repo (pyproject + lock updated). h3ronpy 0.22, duckdb 1.5.5,
icechunk 2.1.2 were already here. Dask is NOT in the stack; it lands in the venv
only as a transitive dep of xarray-sql (its chunk handoff), nothing of ours
imports it.

## Pipeline

1. Open icechunk main, select `cnc_dust`, chain steps 1..24 of the last 6 daily
   runs (2026-08-25..30 today) into a gap-free hourly series, ~144 frames /
   ~1.9 GB read / ~1.5 min sequential. Knob for "forecast mode": today's run,
   steps 1..120 instead (120 frames, same cost shape).
2. Threshold before anything leaves DataFusion: read one chunk, take p99 as
   DUST_MIN (the prior notebook's move), `WHERE cnc_dust > DUST_MIN`. This is
   what makes a 6.4M-cell global grid tractable per frame.
3. H3 in the same SQL pass via an h3ronpy UDF registered on the DataFusion ctx
   (h3ronpy is arrow-native, so lat/lon arrays -> uint64 cells without leaving
   Arrow). Res 4 global (prior dust notebook's choice; res 5 if the hex count
   holds up). Group to (hex, valid_time, avg cnc).
4. Hand the small Arrow table to DuckDB: `valid_time = init_time + to_hours(step)`
   as a real TIMESTAMP (kepler's parquet reader mis-reads bare int64 epochs),
   dust as ug/m3 (x1e9; scientific notation makes kepler's type-analyzer call the
   column STRING).
5. Kontur: `kontur_population_20231101_r6.gpkg.gz` from Kontur's public S3
   (geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/, 177 MB
   gz; r4 also exists at 6.6 MB). NOT yet downloaded. st_read the gpkg, roll up
   ONCE with `h3_cell_to_parent(h3, RES)` summing population, then a plain
   equi-join. LEFT join + COALESCE(pop, 0) keeps ocean/desert plume cells (the
   flat dust-only reading needs them); the extruded reading filters pop > 0 in
   kepler.
6. Write two parquets like before: dust-only and dust+pop.

## Open choices (not decided)

- Window: trailing 6 hindcast days vs today's 120 h forecast vs both as knobs.
- Res 4 vs 5 for the global hexes (row count vs plume texture).
- Kepler vs the repo's own deck.gl anywidget chassis (hex-waves) for the globe;
  kepler was the prior workflow and is where "extrusion = pop, color = dust"
  is a two-click config. deck's SolidPolygonLayer route would animate smoother
  but is a build, not a config.
- A derived `person_dust = pop * cnc` column is one SELECT away if the single
  exposure surface reading is wanted.

## Color (protan-safe)

Dust ramp in kepler: single-hue luminance ramp or cividis-like blue->yellow.
No red-vs-green, no red-alone encodings.
