import icechunk, xarray as xr, numpy as np

st = icechunk.s3_storage(bucket="ctrees-agb-100m-global", prefix="agb_100m_global", region="us-west-2", anonymous=True)
sess = icechunk.Repository.open(st).readonly_session(branch="main")
ct = xr.open_zarr(sess.store, zarr_format=3, group="aboveground_biomass", chunks=None, decode_timedelta=False)
print("CTREES", dict(ct.sizes))
for v in ct.data_vars:
    print(" ", v, ct[v].dtype, ct[v].encoding.get("chunks"), ct[v].encoding.get("preferred_chunks"))
print("  time:", ct.time.values[:3], "...", ct.time.values[-1])

st2 = icechunk.s3_storage(bucket="carbonplan", prefix="carbonplan-ocr/output/fire-risk/tensor/production/v1.1.0/ocr.icechunk",
                          endpoint_url="https://data.source.coop", region="us-west-2", anonymous=True, force_path_style=True)
sess2 = icechunk.Repository.open(st2).readonly_session(branch="main")
ocr = xr.open_zarr(sess2.store, zarr_format=3, chunks=None, decode_timedelta=False)
print("OCR", dict(ocr.sizes))
for v in ocr.data_vars:
    print(" ", v, ocr[v].dtype, ocr[v].encoding.get("chunks"))
la = ocr.latitude.values; lo = ocr.longitude.values
print("  lat", la[0], la[-1], "lon", lo[0], lo[-1], "dlat", la[1]-la[0])
