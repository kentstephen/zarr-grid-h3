"""Cause labels at H3 res 7 from the cboettig hex products on source.coop.
No geometry ops: res-10 cells parent to res 7, and distinct-h10 counts give
burned / harvested fractions (343 res-10 cells per res-7 hex).

MTBS 1984-2024 perimeters: fires with ignition 2001-2024 (the loss window),
split wildfire (incl. Wildland Fire Use) vs prescribed vs unknown.
FACTS common attributes: harvest/thinning activities completed FY2001-2025,
federal land only. The matched activity list is printed for review.

Output: conus/cache/labels_res7.parquet
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "cache" / "labels_res7.parquet"

MTBS = "s3://cboettig/fire/mtbs-perimeters-1984-2024/hex/h0=*/data_0.parquet"
FACTS = "s3://cboettig/facts/common-attributes-2026-06/hex/h0=*/data_0.parquet"

HARVEST = r"clearcut|thin|salvage|shelterwood|seed.tree|selection cut|overstory removal|sanitation|liberation|improvement cut|coppice|patch cut|harvest"
NOT_HARVEST = r"exam|survey|need|prescription|monitor|certif|natural"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs; INSTALL h3 FROM community; LOAD h3;")
con.execute("CREATE SECRET sc (TYPE s3, ENDPOINT 'data.source.coop', URL_STYLE 'path', REGION 'us-west-2', USE_SSL true, PROVIDER config);")
con.execute("SET threads=8;")

print("matched FACTS activities:")
acts = con.execute(f"""
    SELECT ACTIVITY, count(*) n FROM read_parquet('{FACTS}')
    WHERE regexp_matches(lower(ACTIVITY), '{HARVEST}')
      AND NOT regexp_matches(lower(ACTIVITY), '{NOT_HARVEST}')
    GROUP BY 1 ORDER BY n DESC""").fetchall()
for a, n in acts: print(f"  {n:>9,}  {a}")

print("building MTBS aggregate...")
con.execute(f"""
CREATE TABLE mtbs AS
WITH f AS (
  SELECT h3_cell_to_parent(h10, 7) AS h7, h10,
         CASE WHEN Incid_Type IN ('Wildfire', 'Wildland Fire Use') THEN 'wf'
              WHEN Incid_Type = 'Prescribed Fire' THEN 'rx' ELSE 'other' END AS cls,
         year(Ig_Date) AS yr
  FROM read_parquet('{MTBS}')
  WHERE Ig_Date >= DATE '2001-01-01'
)
SELECT h7,
  count(DISTINCT h10) FILTER (cls = 'wf')    AS n10_wf,
  count(DISTINCT h10) FILTER (cls = 'rx')    AS n10_rx,
  count(DISTINCT h10) FILTER (cls = 'other') AS n10_fire_other,
  min(yr) FILTER (cls = 'wf') AS first_ig_wf,
  max(yr) FILTER (cls = 'wf') AS last_ig_wf
FROM f GROUP BY h7
""")
n = con.execute("SELECT count(*) FROM mtbs").fetchone()[0]
print(f"  mtbs hexes: {n:,}")

print("building FACTS aggregate...")
con.execute(f"""
CREATE TABLE facts AS
WITH f AS (
  SELECT h3_cell_to_parent(h10, 7) AS h7, h10,
         COALESCE(FISCAL_YEAR_COMPLETED, year(DATE_COMPLETED)) AS fy
  FROM read_parquet('{FACTS}')
  WHERE regexp_matches(lower(ACTIVITY), '{HARVEST}')
    AND NOT regexp_matches(lower(ACTIVITY), '{NOT_HARVEST}')
    AND COALESCE(FISCAL_YEAR_COMPLETED, year(DATE_COMPLETED)) BETWEEN 2001 AND 2025
)
SELECT h7, count(DISTINCT h10) AS n10_harvest,
       min(fy) AS first_fy_harvest, max(fy) AS last_fy_harvest
FROM f GROUP BY h7
""")
n = con.execute("SELECT count(*) FROM facts").fetchone()[0]
print(f"  facts hexes: {n:,}")

con.execute(f"""
COPY (
  SELECT COALESCE(m.h7, f.h7) AS h3,
         COALESCE(n10_wf, 0) AS n10_wf, COALESCE(n10_rx, 0) AS n10_rx,
         COALESCE(n10_fire_other, 0) AS n10_fire_other,
         first_ig_wf, last_ig_wf,
         COALESCE(n10_harvest, 0) AS n10_harvest, first_fy_harvest, last_fy_harvest
  FROM mtbs m FULL JOIN facts f ON m.h7 = f.h7
) TO '{OUT}' (FORMAT parquet)
""")
print(f"wrote {OUT}: {con.execute(f"SELECT count(*) FROM read_parquet('{OUT}')").fetchone()[0]:,} hexes")
