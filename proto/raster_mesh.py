# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo", "anywidget>=0.9", "numpy==2.5.1", "traitlets"]
# ///
"""PROTOTYPE: the HRRR raster drawn on its own grid, H3 as the transformation.

Nothing H3 is drawn. The map is the HRRR field, one textured quad per 3 km pixel,
on a mesh built from the store's own Lambert grid (pixel corners through pyproj, no
warp, nearest-filter texture). Underneath, every land pixel carries its H3 res 7
label (1.00 pixel per cell, measured: 1,905,141 labels for 1,905,141 pixels) and the
label's res 6 parent. H3 does three things here, none of them visible as hexagons:

  1. PICK: a click becomes latLngToCell(res 7) and the label index names the pixel.
     If the click lands in one of the ~40% of land-area res 7 cells that hold no
     pixel centre, the LCC inverse snaps it to the pixel (the "snap" from docs/02).
  2. DOMES: the sustained-heat threshold is applied per pixel; membership can then
     be decided per pixel (no H3) or per res 6 parent cell (any / majority / all of
     the cell's pixels above the level). The member set is drawn as a PIXEL-EDGE
     outline, never as hexagons. The difference between the two rules is what H3
     adds and removes.
  3. COUNTY: the county name a pick reports comes from the res 6 polyfill join.

Data comes from proto/prep.py (proto/cache/*.npy): a 48 h slab of the eastern heat
dome (Jul 1-2 2026) read from the disk mirror. Run: uv run marimo edit proto/raster_mesh.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import pathlib

    import anywidget
    import marimo as mo
    import numpy as np
    import traitlets

    return anywidget, json, mo, np, pathlib, traitlets


@app.cell
def _(json, np, pathlib):
    CACHE = pathlib.Path(__file__).parent / "cache" if "__file__" in globals() else pathlib.Path("proto/cache")
    corners = np.load(CACHE / "corners_wm.npy")  # (ny+1, nx+1, 2) float32 world coords
    label7 = np.load(CACHE / "label7.npy")  # (ny, nx) uint64
    parent6 = np.load(CACHE / "parent6.npy")
    land = np.load(CACHE / "land.npy")
    county_idx = np.load(CACHE / "county_idx.npy")
    county_names = json.loads((CACHE / "county_names.json").read_text())
    hi_q = np.load(CACHE / "hi_q.npy")  # (F, ny, nx) uint8
    frame_labels = json.loads((CACHE / "labels.json").read_text())
    ny, nx = land.shape
    F = hi_q.shape[0]
    # land pixels only cross the bridge: flat index, label, parent index, county, frames
    lidx = np.flatnonzero(land.ravel()).astype(np.uint32)
    N = lidx.size
    lab = label7.ravel()[lidx]
    par_cells, pidx = np.unique(parent6.ravel()[lidx], return_inverse=True)
    pidx = pidx.astype(np.uint32)
    frames = hi_q.reshape(F, -1)[:, lidx]  # (F, N) uint8
    cty = county_idx.ravel()[lidx]
    hi = np.where(frames == 255, np.nan, frames.astype(np.float32) / 2 - 40)
    mid = float(np.nanmedian(hi))
    span = float(max(mid - np.nanpercentile(hi, 2), np.nanpercentile(hi, 98) - mid))
    stats = (
        f"grid {ny}x{nx} · {N:,} land pixels · {par_cells.size:,} res 6 parents "
        f"({N / par_cells.size:.2f} px per cell) · {F} frames · "
        f"{frames.nbytes / 1e6:.0f} MB frames + {corners.nbytes / 1e6:.0f} MB mesh to the browser"
    )
    return (
        F,
        N,
        corners,
        county_names,
        cty,
        frame_labels,
        frames,
        lab,
        lidx,
        mid,
        nx,
        ny,
        pidx,
        span,
        stats,
    )


@app.cell
def _(anywidget, traitlets):
    class RasterFilm(anywidget.AnyWidget):
        """deck.gl SimpleMeshLayer over the native grid; H3 labels for picking and domes."""

        _esm = r"""
        import {Deck, COORDINATE_SYSTEM} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {SimpleMeshLayer} from "https://esm.sh/@deck.gl/mesh-layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {latLngToCell} from "https://esm.sh/h3-js@4.5.0";

        const CSS = `
          .rf { --panel:rgba(15,18,22,.86); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .rf * { box-sizing: border-box; }
          .rf .rf-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .rf .rf-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .rf .rf-hud { position: absolute; z-index: 5; }
          .rf .rf-tl { top: .6rem; left: .6rem; width: 23rem; max-width: calc(100% - 1.2rem); }
          .rf .rf-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .rf .rf-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .rf .rf-ttl { font-weight: 600; }
          .rf .rf-sub, .rf .rf-dim { color: var(--dim); }
          .rf .rf-sub { display: block; margin-top: .1rem; }
          .rf .rf-fields { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .5rem; }
          .rf button.rf-b, .rf select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .18rem .45rem; cursor: pointer; font: inherit; line-height: 1.2; }
          .rf button.rf-on { background: #3a3f2a; border-color: var(--accent); color: #fff; }
          .rf .rf-legend { display: flex; align-items: center; gap: .45rem; margin-top: .45rem; }
          .rf .rf-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .rf .rf-row { display: flex; justify-content: space-between; gap: .6rem; margin-top: .35rem; }
          .rf .rf-p { display: grid; grid-template-columns: 6.4rem 1fr 3.6rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .rf .rf-p label { color: var(--dim); }
          .rf input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .rf .rf-params { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .rf .rf-transport { display: flex; align-items: center; gap: .55rem; }
          .rf .rf-track { flex: 1; }
          .rf .rf-stamp { font-size: 15px; min-width: 11rem; }
          .rf .rf-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; text-align: right; white-space: pre; }
          .rf .rf-pick { margin-top: .35rem; }
        `;
        function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
        function buildLut(stops) {
          const rgb = stops.map(hexToRgb), lut = new Uint8Array(256 * 3);
          for (let i = 0; i < 256; i++) {
            const t = i / 255 * (rgb.length - 1), k = Math.min(rgb.length - 2, Math.floor(t)), f = t - k;
            for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
          }
          return lut;
        }
        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset ?? 0, v.byteLength);
          return null;
        }
        const typed = (u8, T) => u8 && u8.length ? new T(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength)) : null;
        const HI_OF = q => q / 2 - 40;

        // HRRR's Lambert conformal conic (sphere R=6371229, lat0=lon0 tangent at 38.5/-97.5),
        // forward, for the SNAP: lon/lat -> pixel row/col without any H3. This is what
        // "faithful to the native grid" means in code: the raster's own inverse.
        const LCC = (() => {
          const R = 6371229, lat0 = 38.5 * Math.PI / 180, lon0 = -97.5 * Math.PI / 180;
          const n = Math.sin(lat0), F = Math.cos(lat0) * Math.pow(Math.tan(Math.PI / 4 + lat0 / 2), n) / n;
          const rho0 = R * F / Math.pow(Math.tan(Math.PI / 4 + lat0 / 2), n);
          return (lon, lat) => {
            const la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
            const rho = R * F / Math.pow(Math.tan(Math.PI / 4 + la / 2), n), th = n * (lo - lon0);
            return [rho * Math.sin(th), rho0 - rho * Math.cos(th)];
          };
        })();

        function render({model, el}) {
          el.innerHTML = "";
          const root = document.createElement("div"); root.className = "rf";
          root.innerHTML = `<style>${CSS}</style>
            <div class="rf-map">
              <div class="rf-hud rf-tl"><div class="rf-card">
                <span class="rf-ttl"></span><span class="rf-sub"></span>
                <div class="rf-fields">
                  <button class="rf-b rf-fi rf-on" data-field="index">heat index</button>
                  <button class="rf-b rf-fi" data-field="load">sustained heat</button>
                  <select class="rf-rule" title="dome membership rule">
                    <option value="pixel">dome: per pixel (no H3)</option>
                    <option value="any">dome: res 6 cell, any pixel</option>
                    <option value="majority" selected>dome: res 6 cell, majority</option>
                    <option value="all">dome: res 6 cell, all pixels</option>
                  </select>
                </div>
                <div class="rf-legend"><span class="rf-num rf-lo"></span><div class="rf-grad"></div><span class="rf-num rf-hi"></span></div>
                <div class="rf-row"><span class="rf-dim rf-meank">CONUS mean</span><span class="rf-num rf-mean">–</span></div>
                <div class="rf-pick rf-dim">click a pixel</div>
                <div class="rf-params">
                  <div class="rf-p"><label>half-life</label><input type="range" class="rf-half" min="1" max="72" step="1" value="12"><span class="rf-num rf-halfv"></span></div>
                  <div class="rf-p"><label>threshold</label><input type="range" class="rf-thr" min="15" max="40" step="0.5" value="27"><span class="rf-num rf-thrv"></span></div>
                  <div class="rf-p"><label>dome level</label><input type="range" class="rf-lvl" min="0.5" max="10" step="0.5" value="3"><span class="rf-num rf-lvlv"></span></div>
                </div>
              </div></div>
              <span class="rf-ruler rf-num"></span>
              <div class="rf-hud rf-bl"><div class="rf-card rf-transport">
                <button class="rf-b rf-prev">‹</button><button class="rf-b rf-play">▶</button><button class="rf-b rf-next">›</button>
                <div class="rf-track"><input class="rf-frame" type="range" min="0" max="0" value="0" step="1"></div>
                <div class="rf-stamp rf-num"><span class="rf-stampv">–</span></div>
                <select class="rf-fps"><option>2</option><option>4</option><option selected>6</option><option>8</option><option>12</option></select>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".rf-map"), ruler = q(".rf-ruler"), slider = q(".rf-frame"), stampV = q(".rf-stampv"), playBtn = q(".rf-play"), fpsSel = q(".rf-fps");
          const grad = q(".rf-grad"), loEl = q(".rf-lo"), hiEl = q(".rf-hi"), meanEl = q(".rf-mean"), meanK = q(".rf-meank"), pickEl = q(".rf-pick");
          const halfIn = q(".rf-half"), thrIn = q(".rf-thr"), lvlIn = q(".rf-lvl"), ruleSel = q(".rf-rule");

          let cfg = {}, ny = 0, nx = 0, N = 0, F = 0, P = 0;
          let corners = null, lidx = null, lab = null, pidx = null, cty = null, names = [], frames = null;
          let load = null, loadHi = 1, meansI = null, meansL = null;
          let mesh = null, tex = null, texData = null, lutI = null, lutL = null;
          let labelIndex = new Map(), pixRow = null;   // flat pixel -> land row (Int32, -1)
          let frame = 0, field = "index", playing = false, timer = null, deck = null, selected = -1, gen = 0;
          let pcount = null;                            // pixels per parent cell
          const HOME = {longitude: -84.0, latitude: 37.5, zoom: 4.6, minZoom: 2, maxZoom: 12};

          function loadStatic() {
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            ny = cfg.ny; nx = cfg.nx; N = cfg.n; F = cfg.f; P = cfg.p;
            corners = typed(bytesOf(model.get("corners")), Float32Array);
            lidx = typed(bytesOf(model.get("lidx")), Uint32Array);
            lab = typed(bytesOf(model.get("labels")), BigUint64Array);
            pidx = typed(bytesOf(model.get("pidx")), Uint32Array);
            cty = typed(bytesOf(model.get("cty")), Uint16Array);
            frames = typed(bytesOf(model.get("frames")), Uint8Array);
            try { names = JSON.parse(model.get("names") || "[]"); } catch (e) { names = []; }
            lutI = buildLut(cfg.index_stops); lutL = buildLut(cfg.load_stops);
            mapEl.style.height = (cfg.height || 640) + "px";
            const t0 = performance.now();
            // THE LABEL INDEX: res 7 cell id -> land row. This is the H3 side of the pick.
            labelIndex = new Map();
            for (let i = 0; i < N; i++) labelIndex.set(lab[i].toString(16), i);
            pixRow = new Int32Array(ny * nx).fill(-1);
            for (let i = 0; i < N; i++) pixRow[lidx[i]] = i;
            pcount = new Uint16Array(P);
            for (let i = 0; i < N; i++) pcount[pidx[i]]++;
            // THE MESH: (ny+1)(nx+1) shared corner vertices in world coords, two triangles per pixel,
            // tex coords = pixel grid. Built once; the texture is what changes per frame.
            const V = (ny + 1) * (nx + 1);
            const pos = new Float32Array(V * 3), uv = new Float32Array(V * 2);
            for (let r = 0; r <= ny; r++) for (let c = 0; c <= nx; c++) {
              const v = r * (nx + 1) + c;
              pos[3 * v] = corners[2 * v]; pos[3 * v + 1] = corners[2 * v + 1]; pos[3 * v + 2] = 0;
              uv[2 * v] = c / nx; uv[2 * v + 1] = r / ny;
            }
            const idx = new Uint32Array(ny * nx * 6); let k = 0;
            for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
              const a = r * (nx + 1) + c, b = a + 1, d = a + nx + 1, e = d + 1;
              idx[k++] = a; idx[k++] = d; idx[k++] = b; idx[k++] = b; idx[k++] = d; idx[k++] = e;
            }
            mesh = {attributes: {positions: {value: pos, size: 3}, texCoords: {value: uv, size: 2}}, indices: {value: idx, size: 1}};
            tex = document.createElement("canvas"); tex.width = nx; tex.height = ny;
            texData = tex.getContext("2d").createImageData(nx, ny);
            slider.max = String(Math.max(0, F - 1));
            meansI = new Float32Array(F);
            for (let f = 0; f < F; f++) { let s = 0, n = 0; for (let i = 0; i < N; i++) { const v = frames[f * N + i]; if (v !== 255) { s += HI_OF(v); n++; } } meansI[f] = n ? s / n : NaN; }
            ruler.textContent = `${N.toLocaleString()} land px · ${(ny * nx).toLocaleString()} quads · ${P.toLocaleString()} res 6 cells\nindex + mesh ${Math.round(performance.now() - t0)} ms`;
            q(".rf-ttl").textContent = cfg.title || ""; q(".rf-sub").textContent = cfg.subtitle || "";
            computeLoad(); legend();
          }

          const params = () => ({half: parseFloat(halfIn.value) || 12, thr: parseFloat(thrIn.value) || 27, lvl: parseFloat(lvlIn.value) || 3});
          function computeLoad() {
            const {half, thr} = params(); const a = Math.pow(2, -1 / half), b = 1 - a;
            load = load && load.length === F * N ? load : new Uint8Array(F * N);
            const prev = new Float32Array(N), hist = new Uint32Array(256); meansL = new Float32Array(F);
            for (let f = 0; f < F; f++) {
              const base = f * N; let s = 0, n = 0;
              for (let i = 0; i < N; i++) {
                const k = base + i, qv = frames[k];
                if (qv === 255) { load[k] = 255; continue; }
                let ex = HI_OF(qv) - thr; if (ex < 0) ex = 0;
                const L = a * prev[i] + b * ex; prev[i] = L;
                let lq = Math.round(L * 10); if (lq > 254) lq = 254;
                load[k] = lq; hist[lq]++; s += L; n++;
              }
              meansL[f] = n ? s / n : NaN;
            }
            let tot = 0; for (let i = 1; i < 255; i++) tot += hist[i];
            let acc = 0, top = 10; for (let i = 1; i < 255; i++) { acc += hist[i]; if (acc >= tot * 0.98) { top = i; break; } }
            loadHi = Math.max(1, top / 10); gen++;
          }
          function paramLabels() {
            const p = params();
            q(".rf-halfv").textContent = p.half + " h"; q(".rf-thrv").textContent = p.thr.toFixed(1) + "°C"; q(".rf-lvlv").textContent = "+" + p.lvl.toFixed(1) + "°C";
          }
          let ptimer = null;
          halfIn.oninput = thrIn.oninput = () => { paramLabels(); if (ptimer) clearTimeout(ptimer); ptimer = setTimeout(() => { computeLoad(); legend(); update(); }, 120); };
          lvlIn.oninput = () => { paramLabels(); update(); };
          ruleSel.onchange = () => update();

          // THE TEXTURE: one RGBA image of the grid, land pixels from the field, the rest alpha 0.
          function paintTexture() {
            const d = texData.data; d.fill(0);
            const src = field === "load" ? load : frames, lut = field === "load" ? lutL : lutI, base = frame * N;
            for (let i = 0; i < N; i++) {
              const qv = src[base + i]; if (qv === 255) continue;
              let t;
              if (field === "load") { t = (qv / 10) / loadHi; if (t > 1) t = 1; }
              else { t = (HI_OF(qv) - cfg.lo) / (cfg.hi - cfg.lo); t = t < 0 ? 0 : t > 1 ? 1 : t; }
              const j = Math.round(t * 255) * 3, o = lidx[i] * 4;
              d[o] = lut[j]; d[o + 1] = lut[j + 1]; d[o + 2] = lut[j + 2]; d[o + 3] = field === "load" ? 70 + Math.round(170 * t) : 230;
            }
            tex.getContext("2d").putImageData(texData, 0, 0);
            return tex;
          }

          // THE DOME: per-pixel threshold, then membership by rule, then PIXEL EDGES.
          const member = new Uint8Array(1);
          function domeEdges() {
            if (!load) return null;
            const {lvl} = params(); const lq = Math.round(lvl * 10), rule = ruleSel.value, base = frame * N;
            const above = new Uint8Array(N); let nAbove = 0;
            for (let i = 0; i < N; i++) { const v = load[base + i]; if (v !== 255 && v >= lq) { above[i] = 1; nAbove++; } }
            let mem = above;
            if (rule !== "pixel") {
              // H3 AS THE TRANSFORMATION: count above-pixels per res 6 parent, decide per cell,
              // hand the decision back to every pixel of the cell.
              const cnt = new Uint16Array(P);
              for (let i = 0; i < N; i++) if (above[i]) cnt[pidx[i]]++;
              const cellIn = new Uint8Array(P);
              for (let p = 0; p < P; p++) {
                const c = cnt[p], n = pcount[p];
                cellIn[p] = rule === "any" ? (c > 0 ? 1 : 0) : rule === "all" ? (c === n ? 1 : 0) : (c * 2 > n ? 1 : 0);
              }
              mem = new Uint8Array(N); for (let i = 0; i < N; i++) mem[i] = cellIn[pidx[i]];
            }
            // pixel-edge outline: an edge between a member pixel and a non-member (or non-land) pixel
            const memGrid = new Uint8Array(ny * nx); let nMem = 0;
            for (let i = 0; i < N; i++) if (mem[i]) { memGrid[lidx[i]] = 1; nMem++; }
            const pos = []; const W = nx + 1;
            const push = (v0, v1) => { pos.push(corners[2 * v0], corners[2 * v0 + 1], corners[2 * v1], corners[2 * v1 + 1]); };
            for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
              const g = r * nx + c; if (!memGrid[g]) continue;
              const a = r * W + c;
              if (r === 0 || !memGrid[g - nx]) push(a, a + 1);             // top edge
              if (r === ny - 1 || !memGrid[g + nx]) push(a + W, a + W + 1); // bottom
              if (c === 0 || !memGrid[g - 1]) push(a, a + W);              // left
              if (c === nx - 1 || !memGrid[g + 1]) push(a + 1, a + W + 1);  // right
            }
            return {pos: new Float32Array(pos), nAbove, nMem, E: pos.length / 4};
          }

          const tiles = (id, url, opacity) => new TileLayer({
            id, data: url, tileSize: 256, minZoom: 0, maxZoom: 19, opacity, pickable: false,
            renderSubLayers: p => { const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          let domeInfo = "";
          function layers() {
            const out = [tiles("base", "https://basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}.png", 1.0)];
            if (mesh) {
              out.push(new SimpleMeshLayer({
                id: "raster", data: [0], mesh, texture: paintTexture(),
                coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                getPosition: d => [0, 0, 0], getColor: [255, 255, 255, 255],
                material: false, pickable: false,
                textureParameters: {minFilter: "nearest", magFilter: "nearest"},
                updateTriggers: {texture: [frame, field, gen]},
                parameters: {depthTest: false},
              }));
              const d = domeEdges();
              if (d && d.E) {
                const startIndices = new Uint32Array(d.E + 1); for (let e = 0; e <= d.E; e++) startIndices[e] = 2 * e;
                out.push(new PathLayer({
                  id: "dome", data: {length: d.E, startIndices, attributes: {getPath: {value: d.pos, size: 2}}},
                  _pathType: "open", coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                  getColor: [247, 209, 61, 230], getWidth: 1.5, widthUnits: "pixels", widthMinPixels: 1,
                  jointRounded: false, capRounded: false, pickable: false,
                }));
              }
              domeInfo = d ? `dome ≥ +${params().lvl}°C: ${d.nAbove.toLocaleString()} px above · ${d.nMem.toLocaleString()} member px (${ruleSel.value}) · ${d.E.toLocaleString()} edges` : "";
              if (selected >= 0) {
                const g = lidx[selected], r = Math.floor(g / nx), c = g % nx, a = r * (nx + 1) + c, W = nx + 1;
                const ring = [a, a + 1, a + W + 1, a + W, a].map(v => [corners[2 * v], corners[2 * v + 1]]);
                out.push(new PathLayer({id: "picked", data: [ring], getPath: x => x, coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
                  getColor: [255, 255, 255, 255], getWidth: 2, widthUnits: "pixels", pickable: false}));
              }
            }
            out.push(tiles("labels", "https://basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png", 0.6));
            return out;
          }
          const fmtC = v => Number.isFinite(v) ? v.toFixed(1) + "°C" : "no data";
          function legend() {
            const lut = field === "load" ? lutL : lutI; const stops = [];
            for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            if (field === "load") { loEl.textContent = "0"; hiEl.textContent = "+" + loadHi.toFixed(1) + "°C"; meanK.textContent = "CONUS mean sustained heat"; }
            else { loEl.textContent = fmtC(cfg.lo); hiEl.textContent = fmtC(cfg.hi); meanK.textContent = "CONUS mean heat index"; }
            root.querySelectorAll(".rf-fi").forEach(b => b.classList.toggle("rf-on", b.dataset.field === field));
          }
          root.querySelectorAll(".rf-fi").forEach(b => { b.onclick = () => { field = b.dataset.field; legend(); update(); }; });
          const valAt = (f, i) => { const s = field === "load" ? load : frames; const v = s[f * N + i]; return v === 255 ? NaN : field === "load" ? v / 10 : HI_OF(v); };
          function update() {
            if (!deck) return;
            deck.setProps({layers: layers()});
            slider.value = String(frame); stampV.textContent = cfg.labels?.[frame] ?? `frame ${frame}`;
            const m = field === "load" ? meansL : meansI; meanEl.textContent = m ? (field === "load" ? "+" : "") + fmtC(m[frame]) : "–";
            if (selected >= 0) pickEl.innerHTML = pickEl.dataset.how + ` · <span class="rf-num">${(field === "load" ? "+" : "") + fmtC(valAt(frame, selected))}</span>`;
            ruler.textContent = ruler.textContent.split("\n").slice(0, 2).join("\n") + "\n" + domeInfo;
          }
          // THE PICK: H3 first (latLngToCell res 7 -> label index), LCC snap when the cell is empty.
          function pick(lng, lat) {
            let i = -1, how = "";
            try { i = labelIndex.get(latLngToCell(lat, lng, 7)) ?? -1; } catch (e) { i = -1; }
            if (i >= 0) how = "via H3 label (res 7)";
            else {
              const [x, y] = LCC(lng, lat);
              const c = Math.floor((x - cfg.x0) / cfg.dx), r = Math.floor((cfg.y0 - y) / cfg.dy);
              if (r >= 0 && r < ny && c >= 0 && c < nx) { i = pixRow[r * nx + c]; how = i >= 0 ? "via LCC snap (empty res 7 cell)" : "not land"; } else how = "outside the grid";
            }
            if (i >= 0) {
              const g = lidx[i], r = Math.floor(g / nx), c = g % nx;
              const cn = cty && cty[i] !== 65535 ? names[cty[i]] : "no county";
              selected = i; pickEl.dataset.how = `px (${r}, ${c}) · ${cn}<br>${how} · cell ${lab[i].toString(16)}`;
            } else { selected = -1; pickEl.textContent = how; pickEl.dataset.how = ""; }
            update();
          }
          function setPlaying(p) {
            playing = p; playBtn.textContent = p ? "❚❚" : "▶";
            if (timer) { clearInterval(timer); timer = null; }
            if (p && F > 1) timer = setInterval(() => { frame = (frame + 1) % F; update(); }, 1000 / (parseFloat(fpsSel.value) || 6));
          }
          playBtn.onclick = () => setPlaying(!playing);
          q(".rf-prev").onclick = () => { frame = (frame - 1 + F) % F; update(); };
          q(".rf-next").onclick = () => { frame = (frame + 1) % F; update(); };
          slider.oninput = () => { frame = parseInt(slider.value) || 0; update(); };
          fpsSel.onchange = () => { if (playing) setPlaying(true); };
          root.tabIndex = 0;
          root.addEventListener("keydown", ev => {
            if (["INPUT", "SELECT", "BUTTON"].includes(ev.target.tagName)) return;
            if (ev.key === " ") { ev.preventDefault(); setPlaying(!playing); }
            else if (ev.key === "ArrowLeft") { ev.preventDefault(); q(".rf-prev").click(); }
            else if (ev.key === "ArrowRight") { ev.preventDefault(); q(".rf-next").click(); }
          });
          function boot() {
            loadStatic(); paramLabels();
            deck = new Deck({parent: mapEl, initialViewState: HOME, controller: true, layers: layers(),
              onError: e => { ruler.textContent = "deck: " + (e && e.message ? e.message : e); console.error(e); }});
            let down = null;
            mapEl.addEventListener("pointerdown", ev => { down = ev.target.closest(".rf-hud") ? null : [ev.clientX, ev.clientY]; }, true);
            mapEl.addEventListener("pointerup", ev => {
              if (!down) return; const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !deck) return;
              const r = mapEl.getBoundingClientRect();
              try { const ll = deck.getViewports()[0].unproject([ev.clientX - r.left, ev.clientY - r.top]); pick(ll[0], ll[1]); }
              catch (e) { ruler.textContent = "unproject: " + e.message; }
            }, true);
            update();
          }
          model.on("change:frames", () => { loadStatic(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (deck) deck.finalize(); };
        }
        export default {render};
        """
        corners = traitlets.Bytes(b"").tag(sync=True)
        lidx = traitlets.Bytes(b"").tag(sync=True)
        labels = traitlets.Bytes(b"").tag(sync=True)
        pidx = traitlets.Bytes(b"").tag(sync=True)
        cty = traitlets.Bytes(b"").tag(sync=True)
        frames = traitlets.Bytes(b"").tag(sync=True)
        names = traitlets.Unicode("[]").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)

    return (RasterFilm,)


@app.cell
def _(
    F,
    N,
    RasterFilm,
    corners,
    county_names,
    cty,
    frame_labels,
    frames,
    json,
    lab,
    lidx,
    mid,
    mo,
    nx,
    ny,
    pidx,
    span,
    stats,
):
    film = mo.ui.anywidget(
        RasterFilm(
            corners=corners.astype("<f4").tobytes(),
            lidx=lidx.astype("<u4").tobytes(),
            labels=lab.astype("<u8").tobytes(),
            pidx=pidx.astype("<u4").tobytes(),
            cty=cty.astype("<u2").tobytes(),
            frames=frames.astype("u1").tobytes(),
            names=json.dumps(county_names),
            config=json.dumps(
                {
                    "ny": int(ny), "nx": int(nx), "n": int(N), "f": int(F), "p": int(pidx.max()) + 1,
                    "x0": -2699020.142521929, "y0": 1588193.847443335, "dx": 3000.0, "dy": 3000.0,
                    "lo": mid - span, "hi": mid + span,
                    "index_stops": ["#08306b", "#2f79b5", "#9ecae1", "#f2f0e6", "#fee391", "#fdb034", "#d94801"],
                    "load_stops": ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d", "#fcffa4"],
                    "labels": frame_labels,
                    "height": 640,
                    "title": "HRRR heat index on its own 3 km grid",
                    "subtitle": "eastern dome, Jul 1-2 2026 · H3 res 7 label per pixel underneath, nothing hexagonal drawn",
                }
            ),
        )
    )
    mo.vstack([film, mo.md(f"<span style='color:#8b929c;font-size:.85em'>{stats}</span>")])
    return (film,)


if __name__ == "__main__":
    app.run()
