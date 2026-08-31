# 08: Biomass change vs burn probability, CONUS

Soft goal, confirmed in a pilot: test whether CTrees biomass change lines up with
CarbonPlan's modeled burn probability. Pilot says the signal exists and is
inverted. Now scale to CONUS and decompose the loss by cause.

## Decisions already made

- No bivariate encodings. Univariate maps, side-by-side panels, or chart+map.
- Full CONUS (OCR's extent), not a regional window.
- Access path is anonymous icechunk on S3/source.coop everywhere. No Earthmover
  account, no GEE.
- Join fabric is H3. Pilot used res 7 (~5 km2); CONUS res is an open choice
  (see below).
- Biomass change metric in the pilot: mean(2023..2025) minus mean(2001..2003),
  Mg/ha. Endpoint 3-year means, not single years.

## Pilot result (Dixie window, must survive the CONUS scale-up to be a story)

Window lon -122..-120.3, lat 39.6..40.6, H3 res 7, 3,181 joined hexes,
forested subset = early AGB >= 50 Mg/ha (n=2,123).

- Spearman rho(bp_2011, delta_agb) = +0.26 forested (p ~ 1e-33): the hexes the
  fire model rated SAFEST lost the most biomass. Lowest-bp quintile median
  -28.6 Mg/ha vs -5.7 for mid quintiles.
- Not explained by starting stock: rho(early_agb, bp) = 0.09, partial rho
  controlling early AGB = 0.24. All three bp formulations agree
  (bp_2011_riley 0.25, rps_scott 0.40).
- Decomposition: inside a rough Dixie bounding box the bp gradient has zero
  power (rho 0.04); outside it the inversion is strong (rho 0.48, likely Camp
  Fire, North Complex fringes, plus private-timber harvest).
- Interpretation so far: bp_2011's gradient describes average-year fire under
  ~2011 climate. The 2018-2021 megafires burned across that gradient, and
  harvest is invisible to it. Caveat: some low-bp loss is logging, which no
  fire model should be blamed for. That is what MTBS/FACTS fix.

Pilot artifacts copied into the repo:
- `proto/signal_check_dixie.py` (runnable end to end, ~20 s)
- `join/cache/hex_join_res7_dixie.parquet` (h3, delta_agb, early_agb, bp_2011)

## Datasets and exact access

### CTrees global AGB 100m, annual 2000-2025 (CC-BY 4.0)

Values are stored scaled: divide by 10 for Mg/ha. Nodata -9999. Global grid,
y descending 90..-90, x ascending -180..180, dims (26, 202500, 405000).
Chunks (1, 4000, 4000): one year per chunk, spatial windows are cheap.
Variables: `agb`, `uncertainty` (same units; grows with biomass, saturation).

```python
import icechunk, xarray as xr
st = icechunk.s3_storage(bucket="ctrees-agb-100m-global",
                         prefix="agb_100m_global",
                         region="us-west-2", anonymous=True)
sess = icechunk.Repository.open(st).readonly_session(branch="main")
ds = xr.open_zarr(sess.store, zarr_format=3,
                  group="aboveground_biomass", chunks=None)
```

### CarbonPlan OCR fire risk v1.1.0, ~30m CONUS (CC-BY 4.0)

Dims (latitude 97579 ascending 22.43..52.48, longitude 208881 -128.4..-64.05).
Variables: `bp_2011`, `bp_2047`, `bp_2011_riley`, `bp_2047_riley`,
`rps_2011`, `rps_2047`, `rps_scott`, `crps_scott`. Snapshots, not a series.

```python
st = icechunk.s3_storage(
    bucket="carbonplan",
    prefix="carbonplan-ocr/output/fire-risk/tensor/production/v1.1.0/ocr.icechunk",
    endpoint_url="https://data.source.coop",
    region="us-west-2", anonymous=True, force_path_style=True)
sess = icechunk.Repository.open(st).readonly_session(branch="main")
ds = xr.open_zarr(sess.store, zarr_format=3, chunks=None)
```

Note: env has no dask, so `chunks=None` (or add dask).

### cboettig/fire on source.coop: burn scars, already H3-indexed (CC-BY)

s3 endpoint `https://data.source.coop`, bucket `cboettig`. Key products:

- `fire/mtbs-perimeters-1984-2024/hex/h0=<res0 cell>/data_0.parquet`
  Hive-partitioned by res-0 parent. Columns include `h10:uint64` (res-10 H3),
  `Event_ID`, `Incid_Name`, `Incid_Type` (wildfire vs prescribed),
  `Ig_Date`, `BurnBndAc`, dNBR severity thresholds.
- `fire/fired-events-2001-2021/hex/...` same layout, adds spread rate/duration.
- Also flat GeoParquet + PMTiles versions of everything, CalFire 1878-2025,
  USGS combined fires, FPA-FOD ignitions, WHP COGs.

### cboettig/facts: USFS harvest/treatment records (federal land only)

- `facts/common-attributes-2026-06/hex/h0=<res0>/data_0.parquet`, res-10 H3,
  7.3M records, `ACTIVITY_CODE`/`ACTIVITY`, `METHOD`, `FISCAL_YEAR_*`,
  `GIS_ACRES`. Filter to harvest/thinning activity codes.

## CONUS pipeline

1. **CTrees ingest.** Read 6 years (2001-03, 2023-25) over the OCR bbox
   (lon -128.4..-64.05, lat 22.43..52.48). At 100m that is roughly
   34k x 72k pixels per year: ~2.4 Gpx window, int16. Process in tiles
   aligned to the (4000, 4000) chunk grid; per tile compute early/late means,
   delta, then hex-aggregate and discard pixels. Also aggregate `uncertainty`
   (late years) for the saturation mask.
2. **OCR ingest.** `bp_2011` (plus `rps_scott` for robustness) over the same
   tiles, subsampled ~3x (pilot showed hex means at 100m sampling are fine).
3. **Hex aggregation.** `h3ronpy.vector.coordinates_to_cells` on pixel
   centers, group means via np.unique/bincount (pilot pattern), or push
   through duckdb. One row per hex: delta_agb, early_agb, mean_unc, bp_2011,
   rps_scott.
4. **Cause labels.** DuckDB over the hive hex parquet:
   `h3_cell_to_parent(h10, R)` from MTBS (burned, ig year, severity,
   wildfire-only filter on Incid_Type) and FACTS (harvested, fiscal year).
   No geometry ops anywhere.
5. **Decomposition + stats.** Per hex: burned / harvested / both / neither.
   Redo the quintile table and rank correlations per class, CONUS-wide and
   by region. The pilot's claim becomes testable properly: does bp separate
   losses among hexes that actually burned, and is the unburned-unharvested
   residual noise or signal?
6. **Viz.** Univariate only. Candidates from earlier discussion: loss map
   with MTBS perimeters as linework; the bp-quintile vs delta chart next to
   it; expected-loss surface as a second map. Palette: protan-safe, no
   red-vs-green, diverging blue/orange for delta.

## Open choices (decide at build time)

- H3 resolution for CONUS: res 6 (~36 km2) keeps the table ~200k hexes and
  draws fast; res 7 (~5 km2, pilot) is ~1.5M hexes. MTBS/FACTS come at res 10
  and parent cleanly to either.
- Whether `bp_2047` enters: comparing bp_2011 vs bp_2047 over the megafire
  footprints answers "did CarbonPlan's own climate adjustment move risk
  toward where the fires happened." One extra column, no new plumbing.
- Forested threshold (pilot 50 Mg/ha early AGB) and big-loss threshold.
- Whether prescribed burns (MTBS Incid_Type) count as "explained" loss.

## Watch-outs carried from the pilot

- CTrees scale factor: raw values are 10x Mg/ha. Divide.
- CTrees saturation: uncertainty grows with biomass; consider graying out
  hexes where mean uncertainty is a large fraction of AGB.
- CTrees is a smoothed ML product: stand-clearing events can appear as
  multi-year ramps, so single-year deltas undercount fresh disturbance.
- bp values are annual probabilities (~0.001-0.04 in pilot window). Do not
  read them as 20-year risk without converting.
- FACTS covers federal land only. Private timber harvest (a chunk of the
  pilot's low-bp loss) stays unlabeled: "neither" class is not "no cause."

---

## Status 2026-08-30: paused, and recentered (read this first)

The CONUS pipeline above was built this session, validated, launched, and then
stopped on purpose. Decision: this plan is the wrong genre for the repo. It is a
batch ETL job feeding a stats report, and the repo is live notebooks over
anonymous S3 that other people can rerun quickly. CTrees is deprioritized:
country-wide at 100m it cannot be a live notebook read (see costs below), and
the project is not attached to CTrees or to the fire story. The concept to
carry forward is the method: joining rasters on different grid systems (and
vector fabrics) through H3, from source.coop / open data, in Stephen's
notebook style (marimo + anywidget, prep script -> cache -> live view, one
thing per moving map, no bivariate, measured docs). What the story becomes is
open and is Stephen's to think about.

### What exists on disk (all under conus/, none committed)

- `conus/ingest_ctrees.py`, `conus/ingest_ocr.py`: tiled, checkpointed,
  chunk-aligned ingests with `--test` (Dixie window) and shard args. Both work.
- `conus/reduce_join.py`, `conus/decompose.py`, `conus/viz.py`: merge + stats +
  matplotlib panels. Written and syntax-checked; decompose/viz never ran on
  real CONUS data. viz.py is matplotlib, off-style, treat as scratch.
- `conus/cache/labels_res7.parquet`: FINISHED and CONUS-wide. 223,291 res-7
  hexes with MTBS wildfire/rx res-10 cell counts (ignition 2001+) and FACTS
  completed harvest/thin counts (FY2001-2025). Reusable for any window work.
- Partial tile caches (`conus/cache/{ctrees,ocr}_tiles/`): CTrees ~4/50, OCR
  ~70/204 when stopped. Safe to delete.

### Validation that did run

Rebuilt the Dixie window through the new tiled path: forested rho(bp_2011,
delta) = +0.286 on 4,507 hexes (chunk-snapped window, so larger than the exact
pilot bbox), consistent with the pilot's +0.26. Pipeline logic is sound.

### Measured costs and corrections to the plan above

- CTrees chunks are (1, 2000, 2000), not (4000, 4000). OCR chunks (6000, 4500).
- CTrees CONUS, 6 years + uncertainty: ~2h serial as written (~0.5s latency per
  chunk read, ~6,600 reads). This is implementation, not physics: compressed
  volume is maybe 10-15 GB, so with real concurrency (obstore fan-out, 100+
  parallel range requests) it is plausibly minutes, and ~1 min from us-west-2.
  Never benchmarked. Window-scale stays cheap: Dixie, all slices, ~20s.
- OCR full CONUS was on pace for ~13 min (204 tile groups).
- MTBS/FACTS hex layout: `h0=<decimal uint64>` partitions (not hex strings);
  activity filter regex and the matched-activity list are in `conus/labels.py`.
- DuckDB h3 community extension works (`h3_cell_to_parent`).

### Method findings worth keeping (the actual yield of the session)

1. Cross-publisher H3 join: CTrees (S3) x CarbonPlan (source.coop) x cboettig
   (source.coop) joined with zero geometry ops. Raster labels parent DOWN from
   pixel centres, cboettig's polygon products parent UP from their res-10
   cells, meeting at res 7 in DuckDB. Three publishers, no shared grid, still
   an equi-join. This is the ecosystem argument for the label layer.
2. New join regime: all prior joins here were 1:1 (similar grids, unique at
   res 7). 100m x 30m is many-to-one: the label is a group key, not an
   identity, and r is chosen after the fact. The other half of the docs/02
   taxonomy, unexercised until now.
3. CTrees at 100m goes unique around res 10, the same resolution cboettig's
   fire cells live at. One dataset, both regimes: fold at 7 or relabel 1:1 at
   10 against someone else's h10 fabric.
4. The unbuilt r_C from docs/02 (H3-driven chunk selection over icechunk) is
   exactly what would make "fly a window" scale-free. Still unbuilt.

### Superseded

The "CONUS pipeline" and "Open choices" sections above are superseded by this
note. Do not resume the CONUS ingest without deciding the genre question
first.
