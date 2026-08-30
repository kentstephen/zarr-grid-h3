# /// script
# requires-python = ">=3.13"
# dependencies = ["marimo", "anywidget", "numpy", "pyarrow", "h3ronpy", "matplotlib"]
# ///
"""Hex waves: burn probability is the height, Fosberg is the colour, the hour rolls.

One set of H3 cells (res 6 by default, res 7 = the label layer itself) carries two
datasets that never left their own grids. HEIGHT is USFS burn probability (Dillon et
al. 2023, FSim, 270 m Albers) grouped into the cell: ~500 pixels per res 6 cell, ~70
per res 7 cell, mean or max (join/prep_bp.py, join/hexagg.py). COLOUR is the Fosberg
Fire Weather Index per hour from Dynamical's HRRR analysis (2 m T and RH, 10 m wind, 3
km Lambert), one FFWI per pixel-hour, grouped into the cell the same way
(join/prep_ffwi.py). Every cell rests at a low fraction of its burn-probability
height; where the hour's FFWI runs from contact toward the colour top the cell
lifts to full height. The lift is disk-smoothed, then unsharped so the front
breaks at its leading edge like surf, and a slow ambient swell rolls the whole
surface. The motion is visual only, not a quantity.
The browser tweens between hours so a front crosses the hexes as a wave.

Run: uv run marimo edit hex-waves.py   (or: uv run python fly_hex.py for screenshots)
Prep: uv run python join/prep_bp.py; uv run python join/prep_ffwi.py 2026-06-29 2026-07-05
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import pathlib
    import sys

    import anywidget
    import marimo as mo
    import numpy as np
    import traitlets
    from matplotlib import colormaps

    ROOT = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
    sys.path.insert(0, str(ROOT / "join"))
    import hexagg

    return ROOT, anywidget, colormaps, hexagg, json, mo, np, traitlets


@app.cell
def _(ROOT):
    # ------------------------------------------------------------------ the view
    RES_H = 6  # 6: ~211k cells, mean/max both meaningful for FFWI (~4 px per cell). 7: the label, 1.48M cells, heavy.
    WINDOW = ("2026-06-29", "2026-07-05")  # a prep_ffwi window on disk
    WINDOW_NPZ = ROOT / "join" / "cache" / f"ffwi_{WINDOW[0]}_{WINDOW[1]}.npz"
    STAT = "mean"  # opening toggle: mean | max (applies to both height and colour)
    HEIGHT_M = 150_000.0  # opening extrusion of the tallest cell, metres (the slider)
    SCALE = "log"  # opening height scale: log | linear | rank
    FFWI_HI = 60.0  # colour ramp top; FFWI >= this is the last viridis stop
    CONTACT = 20.0  # the lift starts here; full height at FFWI_HI
    REST = 0.15  # resting height, every cell's share of its full height
    SMOOTH = 2  # gridDisk(1) averaging rounds on the lift
    CREST = 1.5  # unsharp gain on the lift: the front overshoots at its leading edge and settles behind
    SWELL = 0.10  # ambient travelling swell amplitude, as a share of full height
    FPS = 12  # hours per second when playing; the tween fills between
    BASE_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"  # CARTO Dark Matter vector, no labels, no key
    return (
        BASE_STYLE,
        CONTACT,
        CREST,
        FFWI_HI,
        REST,
        SMOOTH,
        SWELL,
        FPS,
        HEIGHT_M,
        RES_H,
        SCALE,
        STAT,
        WINDOW,
        WINDOW_NPZ,
    )


@app.cell
def _(anywidget, traitlets):
    class HexWaves(anywidget.AnyWidget):
        """deck.gl SolidPolygonLayer over H3 cells: burn probability sets each cell's
        full height, every cell rests at a low fraction of it, and the hour's FFWI
        lifts it the rest of the way (gridDisk-smoothed). Coloured by FFWI, tweened.

        Kernel -> browser: `xy` (float32 V x 2 lon/lat, closed rings), `starts` (uint32
        K+1 ring starts), `cells` (uint64 K), `bp_mean` / `bp_max` (float32 K), `ffwi_mean`
        / `ffwi_max` (uint8 F x K, FFWI in 0.5 steps, 255 = no data), `nbrs` (uint32
        K x 7 gridDisk(1) indices, self-padded), `config` (JSON: times, lut,
        defaults). Nothing goes back.
        """

        _esm = r"""
        import {SolidPolygonLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import maplibregl from "https://esm.sh/maplibre-gl@4.7.1";

        const CSS = `
          .hw { --panel:rgba(15,18,22,.84); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .hw * { box-sizing: border-box; }
          .hw .hw-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .hw .hw-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .hw .hw-map:fullscreen { height: 100vh !important; width: 100vw; }
          .hw .hw-ml { position: absolute; inset: 0; }
          .hw .hw-hud { position: absolute; z-index: 5; }
          .hw .hw-hud.hw-tl { top: .6rem; left: .6rem; width: 22rem; max-width: calc(100% - 1.2rem); }
          .hw .hw-hud.hw-bl { left: .6rem; right: .6rem; bottom: .6rem; }
          .hw .hw-card { background: var(--panel); border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(6px); padding: .5rem .65rem; }
          .hw .hw-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; }
          .hw .hw-ttl { font-weight: 600; }
          .hw .hw-sub { color: var(--dim); display: block; margin-top: .1rem; }
          .hw .hw-fields { display: flex; gap: .3rem; margin-top: .5rem; }
          .hw .hw-fields button { flex: 1; }
          .hw button.hw-b.hw-on { background: #3a3f2a; border-color: var(--accent); color: #fff; }
          .hw .hw-legend { display: flex; align-items: center; gap: .45rem; margin-top: .45rem; }
          .hw .hw-grad { height: .55rem; flex: 1; border: 1px solid rgba(255,255,255,.12); }
          .hw .hw-cap { margin-top: .25rem; }
          .hw.hw-collapsed .hw-body, .hw.hw-collapsed .hw-sub { display: none; }  /* hide folds the panel; the stat switch and the ramp stay */
          .hw .hw-toggle { background: none; border: 0; color: var(--dim); cursor: pointer; font: inherit; padding: 0 .1rem; }
          .hw .hw-toggle:hover { color: var(--ink); }
          .hw .hw-params { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); }
          .hw .hw-p { display: grid; grid-template-columns: 6.2rem 1fr 3.4rem; align-items: center; gap: .4rem; margin-top: .2rem; }
          .hw .hw-p label { color: var(--dim); }
          .hw .hw-p .hw-num { text-align: right; }
          .hw .hw-pick { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid rgba(255,255,255,.08); min-height: 2.6em; }
          .hw .hw-hint { margin-top: .35rem; }
          .hw .hw-transport { display: flex; align-items: center; gap: .55rem; }
          .hw .hw-stamp { font-size: 15px; min-width: 11.5rem; }
          .hw .hw-stamp small { display: block; font-size: 10px; color: var(--dim); letter-spacing: .04em; text-transform: uppercase; }
          .hw .hw-track { flex: 1 1 10rem; position: relative; padding-top: 6px; }
          .hw .hw-ticks { position: absolute; left: 0; right: 0; top: 0; height: 6px; }
          .hw .hw-ticks i { position: absolute; top: 0; width: 1px; height: 6px; background: var(--dim); }
          .hw input[type=range] { width: 100%; margin: 0; accent-color: var(--accent); }
          .hw button.hw-b, .hw select { background: #22282f; color: var(--ink); border: 1px solid #343b45; padding: .22rem .5rem; cursor: pointer; font: inherit; line-height: 1.2; min-width: 2rem; }
          .hw button.hw-b:hover, .hw select:hover { background: #2b323b; }
          .hw button:focus-visible, .hw select:focus-visible, .hw input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
          .hw .hw-dim { color: var(--dim); }
          .hw .hw-ruler { position: absolute; right: .6rem; top: .6rem; color: var(--dim); z-index: 5; }
          @media (max-width: 720px) { .hw .hw-stamp { min-width: 0; } .hw .hw-hud.hw-tl { width: calc(100% - 1.2rem); } }
        `;

        function render({model, el}) {
          if (!document.querySelector("link[data-hw-ml]")) {
            const l = document.createElement("link"); l.rel = "stylesheet";
            l.href = "https://esm.sh/maplibre-gl@4.7.1/dist/maplibre-gl.css"; l.dataset.hwMl = "1"; document.head.appendChild(l);
          }
          const root = document.createElement("div"); root.className = "hw";
          root.innerHTML = `<style>${CSS}</style>
            <div class="hw-map" style="height:760px">
              <div class="hw-ml"></div>
              <div class="hw-hud hw-tl"><div class="hw-card hw-panel">
                <div class="hw-head"><span><span class="hw-ttl"></span><span class="hw-sub"></span></span><button class="hw-toggle" title="hide / show (H)">hide</button></div>
                <div class="hw-fields"><button class="hw-b hw-st hw-on" data-st="mean" title="mean per cell, height and colour">mean</button><button class="hw-b hw-st" data-st="max" title="max per cell, height and colour">max</button></div>
                <div class="hw-legend"><span class="hw-num hw-lo">0</span><div class="hw-grad"></div><span class="hw-num hw-hiv"></span></div>
                <div class="hw-dim hw-cap">colour: Fosberg fire weather this hour &middot; height: burn probability, resting low, lifted where FFWI arrives</div>
                <div class="hw-body">
                  <div class="hw-params">
                    <div class="hw-p"><label>height</label><input type="range" class="hw-h" min="0" max="400" step="5" title="extrusion of the tallest cell"><span class="hw-num hw-hv"></span></div>
                    <div class="hw-p"><label>scale</label><select class="hw-scale" title="burn probability to height"><option value="log">log</option><option value="linear">linear</option><option value="rank">rank</option></select><span></span></div>
                    <div class="hw-p"><label>elev scale</label><input type="range" class="hw-esc" min="0" max="5" step="0.1" value="1" title="multiplier on every column"><span class="hw-num hw-escv"></span></div>
                    <div class="hw-p"><label>contact</label><input type="range" class="hw-con" min="0" max="60" step="1" title="FFWI where the lift starts; full height at the colour top"><span class="hw-num hw-conv"></span></div>
                    <div class="hw-p"><label>resting</label><input type="range" class="hw-rest" min="0" max="1" step="0.05" title="height floor for every cell, as a share of its full height"><span class="hw-num hw-restv"></span></div>
                    <div class="hw-p"><label>smooth</label><input type="range" class="hw-sm" min="0" max="4" step="1" title="gridDisk(1) averaging rounds on the lift"><span class="hw-num hw-smv"></span></div>
                    <div class="hw-p"><label>crest</label><input type="range" class="hw-cr" min="0" max="3" step="0.1" title="overshoot at the leading edge of the front (unsharp on the lift; needs smooth &gt; 0)"><span class="hw-num hw-crv"></span></div>
                    <div class="hw-p"><label>swell</label><input type="range" class="hw-sw" min="0" max="0.5" step="0.02" title="ambient travelling swell, share of full height"><span class="hw-num hw-swv"></span></div>
                    <div class="hw-p"><label>colour top</label><input type="range" class="hw-hi" min="10" max="100" step="5" title="FFWI at the last viridis stop"><span class="hw-num hw-hitv"></span></div>
                    <div class="hw-p"><label>opacity</label><input type="range" class="hw-opac" min="0.2" max="1" step="0.05"><span class="hw-num hw-opacv"></span></div>
                    <div class="hw-p"><label>base map</label><button class="hw-b hw-base hw-on">on</button><span></span></div>
                  </div>
                  <div class="hw-pick hw-dim">click a cell</div>
                  <div class="hw-dim hw-hint">click a cell for its values &middot; space plays &middot; &larr; &rarr; step &middot; H hide &middot; F fullscreen</div>
                </div>
              </div></div>
              <span class="hw-ruler hw-num"></span>
              <div class="hw-hud hw-bl"><div class="hw-card hw-transport">
                <button class="hw-b hw-prev" title="step back (&larr;)">&lsaquo;</button>
                <button class="hw-b hw-play" title="play / pause (space)">&#9654;</button>
                <button class="hw-b hw-next" title="step forward (&rarr;)">&rsaquo;</button>
                <div class="hw-track"><div class="hw-ticks"></div><input class="hw-frame" type="range" min="0" max="0" step="0.05" value="0" aria-label="frame"></div>
                <div class="hw-stamp hw-num"><small>hour (UTC)</small><span class="hw-stampv"></span></div>
                <select class="hw-fps" title="hours per second"><option value="4">4 h/s</option><option value="8">8 h/s</option><option value="12">12 h/s</option><option value="24">24 h/s</option></select>
                <button class="hw-b hw-home" title="home view">home</button>
                <button class="hw-b hw-full" title="fullscreen (F)">&#x26F6;</button>
              </div></div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".hw-map"), ttl = q(".hw-ttl"), sub = q(".hw-sub"), ruler = q(".hw-ruler"), pickEl = q(".hw-pick"),
                scaleSel = q(".hw-scale"), hIn = q(".hw-h"), hV = q(".hw-hv"), escIn = q(".hw-esc"), escV = q(".hw-escv"),
                conIn = q(".hw-con"), conV = q(".hw-conv"), restIn = q(".hw-rest"), restV = q(".hw-restv"), smIn = q(".hw-sm"), smV = q(".hw-smv"),
                crIn = q(".hw-cr"), crV = q(".hw-crv"), swIn = q(".hw-sw"), swV = q(".hw-swv"),
                hiIn = q(".hw-hi"), hiV = q(".hw-hiv"), hitV = q(".hw-hitv"),
                grad = q(".hw-grad"), baseBtn = q(".hw-base"), opacIn = q(".hw-opac"), opacV = q(".hw-opacv"),
                playBtn = q(".hw-play"), slider = q(".hw-frame"), ticks = q(".hw-ticks"), stampV = q(".hw-stampv"),
                fpsSel = q(".hw-fps"), toggleBtn = q(".hw-toggle");

          let cfg = {}, K = 0, F = 0, V = 0, xy = null, starts = null, cells = null, bp = {}, ffwi = {}, data = null;
          let stat = "mean", t = 0, playing = false, raf = 0, lastTs = 0, deck = null, map = null, useBase = true, selected = -1;
          let vcell = null;                 // vertex -> cell index (per-vertex attributes for the polygon layer)
          let color = [null, null], cbuf = 0, elev = [null, null], ebuf = 0, perH = null, rank = null, lut = null, nbrs = null, gA = null, gB = null, gR = null, phase = null;
          let tickMs = 0, upMs = 0;

          const f32 = k => new Float32Array(model.get(k).buffer.slice(0));
          const u32 = k => new Uint32Array(model.get(k).buffer.slice(0));
          const u8 = k => new Uint8Array(model.get(k).buffer.slice(0));
          const u64 = k => new BigUint64Array(model.get(k).buffer.slice(0));

          function buildTicks() {
            ticks.innerHTML = "";
            for (let k = 0; k < F; k++) if (cfg.times[k].endsWith("T00:00")) { const i = document.createElement("i"); i.style.left = (k / (F - 1) * 100) + "%"; ticks.appendChild(i); }
          }

          function loadStatic() {
            cfg = JSON.parse(model.get("config") || "{}");
            xy = f32("xy"); starts = u32("starts"); cells = u64("cells");
            bp = {mean: f32("bp_mean"), max: f32("bp_max")};
            ffwi = {mean: u8("ffwi_mean"), max: u8("ffwi_max")}; nbrs = u32("nbrs");
            K = cells.length; V = xy.length / 2; F = cfg.times.length;
            lut = new Uint8Array(cfg.lut.flat());
            vcell = new Uint32Array(V);
            for (let i = 0; i < K; i++) for (let v = starts[i]; v < starts[i + 1]; v++) vcell[v] = i;
            color = [new Uint8Array(V * 4), new Uint8Array(V * 4)]; elev = [new Float32Array(V), new Float32Array(V)];
            gA = new Float32Array(K); gB = new Float32Array(K); gR = new Float32Array(K);
            // one plane swell across the map: phase from each cell's first vertex, ~2.5 deg wavelength, from the WSW
            phase = new Float32Array(K);
            for (let i = 0; i < K; i++) { const v0 = 2 * starts[i]; phase[i] = (xy[v0] * 0.8 + xy[v0 + 1] * 0.6) * (2 * Math.PI / 2.5); }
            data = {length: K, startIndices: starts, attributes: {getPolygon: {value: xy, size: 2}}};
            stat = cfg.stat; scaleSel.value = cfg.scale; hIn.value = cfg.height_m / 1000; hiIn.value = cfg.ffwi_hi;
            conIn.value = cfg.contact ?? 20; restIn.value = cfg.rest ?? 0.15; smIn.value = cfg.smooth ?? 2; crIn.value = cfg.crest ?? 1.5; swIn.value = cfg.swell ?? 0.1; opacIn.value = cfg.opacity ?? 0.9;
            fpsSel.value = String(cfg.fps); slider.max = String(F - 1); buildTicks();
            ttl.textContent = cfg.title; sub.textContent = cfg.subtitle;
            root.querySelectorAll(".hw-st").forEach(b => b.classList.toggle("hw-on", b.dataset.st === stat));
            rebuildElev(); legend();
          }

          // HEIGHT: bp through the scale, per cell, metres. paint() writes the vertices,
          // gated by the hour's FFWI against the contact threshold.
          function rebuildElev() {
            const b = bp[stat], H = parseFloat(hIn.value) * 1000; hV.textContent = (H / 1000).toFixed(0) + " km";
            const mode = scaleSel.value;
            let f;
            if (mode === "linear") { const mx = cfg.bp_max[stat] || 1; f = i => b[i] / mx; }
            else if (mode === "rank") {
              if (!rank || rank.stat !== stat) { const idx = Array.from({length: K}, (_, i) => i).sort((p, r) => b[p] - b[r]); const rk = new Float32Array(K); idx.forEach((i, r) => rk[i] = r / (K - 1)); rank = {stat, rk}; }
              f = i => rank.rk[i];
            } else { const lo = Math.log10(cfg.bp_floor), hi = Math.log10(cfg.bp_max[stat] || 1); f = i => Math.max(0, (Math.log10(Math.max(b[i], cfg.bp_floor)) - lo) / (hi - lo)); }
            perH = new Float32Array(K); for (let i = 0; i < K; i++) perH[i] = H * f(i);
          }

          // COLOUR + HEIGHT: FFWI at fractional hour t (linear between the two hours), viridis, per
          // vertex. Height rests at `resting` x the cell's burn-probability height and lifts to full
          // as FFWI climbs from contact to the colour top. The lift is averaged over gridDisk(1)
          // `smooth` times, then unsharped by `crest` so the front breaks at its leading edge and
          // settles behind, and a slow plane `swell` (one cycle per 24 h of t) undulates the whole
          // surface. Visual only. No FFWI this hour: alpha 0, flat.
          function paint() {
            const t0 = performance.now(), k = Math.min(F - 1, Math.floor(t)), a = Math.min(1, t - k), k1 = Math.min(F - 1, k + 1);
            const q = ffwi[stat], A = q.subarray(k * K, (k + 1) * K), B = q.subarray(k1 * K, (k1 + 1) * K);
            const hi = parseFloat(hiIn.value) * 2, c = color[cbuf ^= 1], per = new Uint8Array(K * 4);
            const esc = parseFloat(escIn.value), c0 = parseFloat(conIn.value) * 2, rest = parseFloat(restIn.value), S = parseInt(smIn.value);
            const crest = parseFloat(crIn.value), swell = parseFloat(swIn.value), wt = t * (2 * Math.PI / 24);
            const span = Math.max(1, hi - c0), e = elev[ebuf ^= 1];
            let g = gA;
            for (let i = 0; i < K; i++) {
              const x = A[i], y = B[i]; let v;
              if (x === 255 && y === 255) { per[4 * i + 3] = 0; g[i] = 0; gR[i] = 0; continue; }
              v = x === 255 ? y : y === 255 ? x : x + (y - x) * a;
              g[i] = gR[i] = Math.min(1, Math.max(0, (v - c0) / span));
              const j = 3 * Math.min(255, Math.round(v / hi * 255));
              per[4 * i] = lut[j]; per[4 * i + 1] = lut[j + 1]; per[4 * i + 2] = lut[j + 2]; per[4 * i + 3] = 255;
            }
            for (let s = 0; s < S; s++) {
              const h = g === gA ? gB : gA;
              for (let i = 0; i < K; i++) { const b = i * 7; let acc = 0; for (let d = 0; d < 7; d++) acc += g[nbrs[b + d]]; h[i] = acc / 7; }
              g = h;
            }
            const eh = new Float32Array(K);
            for (let i = 0; i < K; i++) {
              const f = g[i] + crest * (gR[i] - g[i]);   // unsharp: crest at the leading edge, trough behind
              eh[i] = perH[i] * esc * Math.max(0, rest + (1 - rest) * f + swell * Math.sin(phase[i] - wt));
            }
            for (let v = 0; v < V; v++) { const i = vcell[v], o = v * 4, j = i * 4; c[o] = per[j]; c[o + 1] = per[j + 1]; c[o + 2] = per[j + 2]; c[o + 3] = per[j + 3]; e[v] = eh[i]; }
            tickMs = performance.now() - t0;
            return c;
          }

          function layers(c) {
            const out = [];
            data.attributes.getFillColor = {value: c, size: 4};
            data.attributes.getElevation = {value: elev[ebuf], size: 1};
            out.push(new SolidPolygonLayer({
              id: "hex", data, _normalize: false, _windingOrder: "CCW", extruded: true, filled: true, wireframe: false,
              opacity: parseFloat(opacIn.value), pickable: true, material: {ambient: 0.45, diffuse: 0.7, shininess: 24, specularColor: [40, 40, 40]},
              updateTriggers: {getFillColor: cbuf, getElevation: ebuf},
            }));
            return out;
          }
          function update() {
            const t0 = performance.now();
            deck.setProps({layers: layers(paint())});
            upMs = performance.now() - t0;
            const k = Math.floor(t), ts = cfg.times[Math.min(F - 1, k)];
            stampV.textContent = ts.replace("T", " "); slider.value = String(t);
            ruler.textContent = `${K.toLocaleString()} res ${cfg.res} cells · ${V.toLocaleString()} vertices · ${F} hours · paint ${tickMs.toFixed(0)} ms, layer ${(upMs - tickMs).toFixed(0)} ms`;
            opacV.textContent = parseFloat(opacIn.value).toFixed(2);
            escV.textContent = parseFloat(escIn.value).toFixed(1) + "x";
            conV.textContent = parseFloat(conIn.value).toFixed(0);
            restV.textContent = parseFloat(restIn.value).toFixed(2); smV.textContent = smIn.value + "x";
            crV.textContent = parseFloat(crIn.value).toFixed(1); swV.textContent = parseFloat(swIn.value).toFixed(2);
          }
          function legend() {
            const stops = []; for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
            const top = parseFloat(hiIn.value); hiV.textContent = top.toFixed(0) + "+"; hitV.textContent = top.toFixed(0);
          }
          function showPick(i) {
            selected = i; if (i < 0) { pickEl.textContent = "click a cell"; return; }
            const k = Math.min(F - 1, Math.round(t)), fm = ffwi.mean[k * K + i], fx = ffwi.max[k * K + i];
            const f = v => v === 255 ? "-" : (v / 2).toFixed(1);
            pickEl.innerHTML = `<span class="hw-num">${cells[i].toString(16)}</span> · burn probability mean <span class="hw-num">${bp.mean[i].toFixed(4)}</span>, max <span class="hw-num">${bp.max[i].toFixed(4)}</span><br>` +
              `FFWI at ${cfg.times[k].slice(5, 16).replace("T", " ")}Z: mean <span class="hw-num">${f(fm)}</span>, max <span class="hw-num">${f(fx)}</span>`;
          }

          // PLAY: t advances fps hours per second, the tween is the fractional part.
          function frame(ts) {
            if (!playing) return;
            if (lastTs) { t += (ts - lastTs) / 1000 * parseFloat(fpsSel.value); if (t >= F - 1) t = 0; }
            lastTs = ts; update(); if (selected >= 0) showPick(selected);
            raf = requestAnimationFrame(frame);
          }
          function setPlaying(p) { playing = p; lastTs = 0; playBtn.textContent = p ? "⏸" : "▶"; playBtn.classList.toggle("hw-on", p); if (p) raf = requestAnimationFrame(frame); else cancelAnimationFrame(raf); }
          playBtn.onclick = () => setPlaying(!playing);
          q(".hw-prev").onclick = () => { setPlaying(false); t = Math.max(0, Math.ceil(t) - 1); update(); if (selected >= 0) showPick(selected); };
          q(".hw-next").onclick = () => { setPlaying(false); t = Math.min(F - 1, Math.floor(t) + 1); update(); if (selected >= 0) showPick(selected); };
          slider.oninput = () => { setPlaying(false); t = parseFloat(slider.value); update(); };
          root.querySelectorAll(".hw-st").forEach(b => b.onclick = () => { stat = b.dataset.st; root.querySelectorAll(".hw-st").forEach(x => x.classList.toggle("hw-on", x === b)); rebuildElev(); update(); if (selected >= 0) showPick(selected); });
          scaleSel.onchange = hIn.oninput = () => { rebuildElev(); update(); };
          escIn.oninput = conIn.oninput = restIn.oninput = smIn.oninput = crIn.oninput = swIn.oninput = opacIn.oninput = () => update();
          hiIn.oninput = () => { legend(); update(); };
          baseBtn.onclick = () => { useBase = !useBase; baseBtn.classList.toggle("hw-on", useBase); baseBtn.textContent = useBase ? "on" : "off"; map.getCanvas().style.visibility = useBase ? "visible" : "hidden"; };
          toggleBtn.onclick = () => { const c = root.classList.toggle("hw-collapsed"); toggleBtn.textContent = c ? "show" : "hide"; };
          q(".hw-home").onclick = () => map.flyTo({center: [cfg.home.longitude, cfg.home.latitude], zoom: cfg.home.zoom, pitch: cfg.home.pitch || 0, bearing: cfg.home.bearing || 0, duration: 600});
          q(".hw-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen(); };
          const onKey = e => {
            if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
            if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
            else if (e.key === "ArrowLeft") q(".hw-prev").click();
            else if (e.key === "ArrowRight") q(".hw-next").click();
            else if (e.key === "h" || e.key === "H") toggleBtn.click();
            else if (e.key === "f" || e.key === "F") q(".hw-full").click();
          };
          document.addEventListener("keydown", onKey);

          function boot() {
            loadStatic();
            map = new maplibregl.Map({
              container: q(".hw-ml"), style: cfg.base_style,
              center: [cfg.home.longitude, cfg.home.latitude], zoom: cfg.home.zoom,
              pitch: cfg.home.pitch || 0, bearing: cfg.home.bearing || 0, maxPitch: 70,
            });
            deck = new MapboxOverlay({
              interleaved: false, layers: [],
              getCursor: ({isHovering}) => isHovering ? "pointer" : "grab",
              onClick: info => showPick(info.index >= 0 && info.layer && info.layer.id === "hex" ? info.index : -1),
            });
            map.addControl(deck);
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:config", () => { loadStatic(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); document.removeEventListener("keydown", onKey); if (map) map.remove(); };
        }
        export default {render};
        """
        xy = traitlets.Bytes(b"").tag(sync=True)
        starts = traitlets.Bytes(b"").tag(sync=True)
        cells = traitlets.Bytes(b"").tag(sync=True)
        bp_mean = traitlets.Bytes(b"").tag(sync=True)
        bp_max = traitlets.Bytes(b"").tag(sync=True)
        ffwi_mean = traitlets.Bytes(b"").tag(sync=True)
        ffwi_max = traitlets.Bytes(b"").tag(sync=True)
        nbrs = traitlets.Bytes(b"").tag(sync=True)
        config = traitlets.Unicode("{}").tag(sync=True)

    return (HexWaves,)


@app.cell
def _(RES_H, WINDOW_NPZ, hexagg, mo, np):
    mo.stop(not WINDOW_NPZ.exists(), mo.md(f"**No window on disk.** Run `uv run python join/prep_ffwi.py {WINDOW_NPZ.stem[5:15]} {WINDOW_NPZ.stem[16:]}` first."))
    hx = hexagg.build(RES_H, WINDOW_NPZ)
    K, F = hx["cells"].size, hx["times"].size
    _bpm = hx["bp"]["mean"]
    build_note = (
        f"res {RES_H}: {K:,} cells, {hx['xy'].shape[0]:,} vertices, {F} hours, built in {hx['build_s']:.1f}s · "
        f"burn probability (mean per cell) p50 {np.percentile(_bpm, 50):.4f}, p90 {np.percentile(_bpm, 90):.4f}, p99 {np.percentile(_bpm, 99):.4f}, max {_bpm.max():.4f} · "
        f"to the browser: {(hx['xy'].nbytes + hx['starts'].nbytes + 2 * hx['ffwi']['mean'].nbytes + 2 * _bpm.nbytes + hx['nbrs'].nbytes + 8 * K) / 1e6:.0f} MB"
    )
    return build_note, hx


@app.cell
def _(
    BASE_STYLE,
    CONTACT,
    CREST,
    FFWI_HI,
    FPS,
    HEIGHT_M,
    HexWaves,
    RES_H,
    REST,
    SCALE,
    SMOOTH,
    STAT,
    SWELL,
    WINDOW,
    colormaps,
    hx,
    json,
    mo,
    np,
):
    _lut = (np.asarray(colormaps["viridis"](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).tolist()
    _lon, _lat = hx["xy"][:, 0], hx["xy"][:, 1]
    _cfg = {
        "res": RES_H, "stat": STAT, "scale": SCALE, "height_m": HEIGHT_M, "ffwi_hi": FFWI_HI, "contact": CONTACT, "rest": REST, "smooth": SMOOTH, "crest": CREST, "swell": SWELL, "fps": FPS, "opacity": 0.9,
        "times": [str(t)[:16] for t in hx["times"]], "lut": _lut,
        "bp_max": {k: float(v.max()) for k, v in hx["bp"].items()}, "bp_floor": 1e-4,
        "base_style": BASE_STYLE, "autoplay": False,
        "home": {"longitude": float((_lon.min() + _lon.max()) / 2), "latitude": float((_lat.min() + _lat.max()) / 2) - 2.0, "zoom": 3.6, "pitch": 55, "bearing": -8},
        "title": "Hex waves: burn probability as height, Fosberg fire weather as colour",
        "subtitle": f"H3 res {RES_H} · USFS FSim burn probability (270 m) x HRRR hourly FFWI (3 km), {WINDOW[0]} to {WINDOW[1]} UTC",
    }
    waves = mo.ui.anywidget(
        HexWaves(
            xy=hx["xy"].astype("<f4").tobytes(), starts=hx["starts"].astype("<u4").tobytes(), cells=hx["cells"].astype("<u8").tobytes(),
            bp_mean=hx["bp"]["mean"].astype("<f4").tobytes(), bp_max=hx["bp"]["max"].astype("<f4").tobytes(),
            ffwi_mean=hx["ffwi"]["mean"].tobytes(), ffwi_max=hx["ffwi"]["max"].tobytes(), nbrs=hx["nbrs"].astype("<u4").tobytes(),
            config=json.dumps(_cfg),
        )
    )
    waves
    return


@app.cell
def _(build_note, mo):
    mo.md(f"""
    `{build_note}`
    """)
    return


if __name__ == "__main__":
    app.run()
