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
(join/prep_ffwi.py). Height is static so time-stepping animates one thing: the colour.
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
    FPS = 12  # hours per second when playing; the tween fills between
    BASE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
    return BASE_TILES, FFWI_HI, FPS, HEIGHT_M, RES_H, SCALE, STAT, WINDOW, WINDOW_NPZ


@app.cell
def _(anywidget, traitlets):
    class HexWaves(anywidget.AnyWidget):
        """deck.gl SolidPolygonLayer over H3 cells: extruded by burn probability, coloured
        by FFWI per hour, tweened.

        Kernel -> browser: `xy` (float32 V x 2 lon/lat, closed rings), `starts` (uint32
        K+1 ring starts), `cells` (uint64 K), `bp_mean` / `bp_max` (float32 K), `ffwi_mean`
        / `ffwi_max` (uint8 F x K, FFWI in 0.5 steps, 255 = no data), `config` (JSON:
        times, lut, defaults). Nothing goes back.
        """

        _esm = r"""
        import {Deck} from "https://esm.sh/@deck.gl/core@9.3.10?deps=apache-arrow@18.1.0";
        import {BitmapLayer, SolidPolygonLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";

        const CSS = `
          .hw { --panel:rgba(15,18,22,.86); --ink:#dfe3e8; --dim:#8b929c; --accent:#e6c14a;
                font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #0f1216; }
          .hw * { box-sizing: border-box; }
          .hw .hw-num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
          .hw .hw-map { position: relative; width: 100%; background: #0b0d10; overflow: hidden; }
          .hw .hw-map:fullscreen { height: 100vh !important; width: 100vw; }
          .hw .hw-hud { position: absolute; z-index: 5; background: var(--panel); border-radius: 6px; padding: .5rem .65rem; backdrop-filter: blur(4px); }
          .hw .hw-tl { top: .6rem; left: .6rem; width: 22rem; max-width: calc(100% - 1.2rem); }
          .hw .hw-bl { left: .6rem; right: .6rem; bottom: .6rem; display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
          .hw .hw-tr { top: .6rem; right: .6rem; width: 16rem; }
          .hw .hw-ttl { font-weight: 600; font-size: 13px; }
          .hw .hw-sub, .hw .hw-dim { color: var(--dim); }
          .hw .hw-row { display: flex; gap: .5rem; align-items: center; margin-top: .35rem; flex-wrap: wrap; }
          .hw .hw-row label { color: var(--dim); min-width: 4.2rem; }
          .hw .hw-b { background: #222831; color: var(--ink); border: 1px solid #3a4250; border-radius: 4px; padding: .15rem .55rem; cursor: pointer; font: inherit; }
          .hw .hw-b.hw-on { background: var(--accent); color: #111; border-color: var(--accent); }
          .hw select, .hw input[type=number] { background: #222831; color: var(--ink); border: 1px solid #3a4250; border-radius: 4px; font: inherit; padding: .1rem .3rem; }
          .hw input[type=range] { flex: 1; min-width: 6rem; accent-color: var(--accent); }
          .hw .hw-frame { flex: 1; min-width: 14rem; }
          .hw .hw-grad { height: 10px; border-radius: 3px; margin-top: .3rem; }
          .hw .hw-lg { display: flex; justify-content: space-between; color: var(--dim); }
          .hw .hw-ruler { color: var(--dim); font-size: 11px; margin-top: .3rem; }
          .hw .hw-pick { margin-top: .35rem; border-top: 1px solid #2a313b; padding-top: .35rem; min-height: 2.6em; }
        `;

        function render({model, el}) {
          const root = document.createElement("div"); root.className = "hw";
          root.innerHTML = `<style>${CSS}</style>
            <div class="hw-map" style="height:760px">
              <div class="hw-hud hw-tl">
                <div class="hw-ttl"></div><div class="hw-sub"></div>
                <div class="hw-row"><label>statistic</label><button class="hw-b hw-st" data-st="mean">mean</button><button class="hw-b hw-st" data-st="max">max</button><span class="hw-dim">per cell, height and colour</span></div>
                <div class="hw-row"><label>height</label><select class="hw-scale"><option value="log">log</option><option value="linear">linear</option><option value="rank">rank</option></select>
                  <input type="range" class="hw-h" min="0" max="400" step="5"><span class="hw-num hw-hv"></span></div>
                <div class="hw-row"><label>colour top</label><input type="number" class="hw-hi" min="10" max="100" step="5" style="width:4.5rem"><span class="hw-dim">FFWI at the last stop</span></div>
                <div class="hw-row"><label>base map</label><button class="hw-b hw-base hw-on">on</button><label>opacity</label><input type="range" class="hw-opac" min="0.2" max="1" step="0.05"><span class="hw-num hw-opacv"></span></div>
                <div class="hw-grad"></div><div class="hw-lg"><span class="hw-lo">0</span><span class="hw-mid">Fosberg fire weather index</span><span class="hw-hiv"></span></div>
                <div class="hw-pick hw-dim">click a cell</div>
                <div class="hw-ruler"></div>
              </div>
              <div class="hw-hud hw-bl">
                <button class="hw-b hw-play">play</button><button class="hw-b hw-prev">&lt;</button><button class="hw-b hw-next">&gt;</button>
                <input type="range" class="hw-frame" min="0" max="0" step="0.05" value="0">
                <span class="hw-num hw-stamp"></span>
                <select class="hw-fps"><option value="4">4 h/s</option><option value="8">8 h/s</option><option value="12">12 h/s</option><option value="24">24 h/s</option></select>
                <button class="hw-b hw-home">home</button><button class="hw-b hw-full">full</button>
              </div>
            </div>`;
          el.appendChild(root);
          const q = s => root.querySelector(s);
          const mapEl = q(".hw-map"), ttl = q(".hw-ttl"), sub = q(".hw-sub"), ruler = q(".hw-ruler"), pickEl = q(".hw-pick"),
                scaleSel = q(".hw-scale"), hIn = q(".hw-h"), hV = q(".hw-hv"), hiIn = q(".hw-hi"), hiV = q(".hw-hiv"), grad = q(".hw-grad"),
                baseBtn = q(".hw-base"), opacIn = q(".hw-opac"), opacV = q(".hw-opacv"),
                playBtn = q(".hw-play"), slider = q(".hw-frame"), stamp = q(".hw-stamp"), fpsSel = q(".hw-fps");

          let cfg = {}, K = 0, F = 0, V = 0, xy = null, starts = null, cells = null, bp = {}, ffwi = {}, data = null;
          let stat = "mean", t = 0, playing = false, raf = 0, lastTs = 0, deck = null, useBase = true, selected = -1;
          let vcell = null;                 // vertex -> cell index (per-vertex attributes for the polygon layer)
          let color = [null, null], cbuf = 0, elev = [null, null], ebuf = 0, rank = null, lut = null;
          let tickMs = 0, upMs = 0;

          const f32 = k => new Float32Array(model.get(k).buffer.slice(0));
          const u32 = k => new Uint32Array(model.get(k).buffer.slice(0));
          const u8 = k => new Uint8Array(model.get(k).buffer.slice(0));
          const u64 = k => new BigUint64Array(model.get(k).buffer.slice(0));

          function loadStatic() {
            cfg = JSON.parse(model.get("config") || "{}");
            xy = f32("xy"); starts = u32("starts"); cells = u64("cells");
            bp = {mean: f32("bp_mean"), max: f32("bp_max")};
            ffwi = {mean: u8("ffwi_mean"), max: u8("ffwi_max")};
            K = cells.length; V = xy.length / 2; F = cfg.times.length;
            lut = new Uint8Array(cfg.lut.flat());
            vcell = new Uint32Array(V);
            for (let i = 0; i < K; i++) for (let v = starts[i]; v < starts[i + 1]; v++) vcell[v] = i;
            color = [new Uint8Array(V * 4), new Uint8Array(V * 4)]; elev = [new Float32Array(V), new Float32Array(V)];
            data = {length: K, startIndices: starts, attributes: {getPolygon: {value: xy, size: 2}}};
            stat = cfg.stat; scaleSel.value = cfg.scale; hIn.value = cfg.height_m / 1000; hiIn.value = cfg.ffwi_hi; opacIn.value = cfg.opacity ?? 0.9;
            fpsSel.value = String(cfg.fps); slider.max = String(F - 1);
            ttl.textContent = cfg.title; sub.textContent = cfg.subtitle;
            root.querySelectorAll(".hw-st").forEach(b => b.classList.toggle("hw-on", b.dataset.st === stat));
            rebuildElev(); legend();
          }

          // HEIGHT: bp through the scale, per vertex, metres. Static until a control moves.
          function rebuildElev() {
            const b = bp[stat], H = parseFloat(hIn.value) * 1000; hV.textContent = (H / 1000).toFixed(0) + " km";
            const mode = scaleSel.value, e = elev[ebuf ^= 1];
            let f;
            if (mode === "linear") { const mx = cfg.bp_max[stat] || 1; f = i => b[i] / mx; }
            else if (mode === "rank") {
              if (!rank || rank.stat !== stat) { const idx = Array.from({length: K}, (_, i) => i).sort((p, r) => b[p] - b[r]); const rk = new Float32Array(K); idx.forEach((i, r) => rk[i] = r / (K - 1)); rank = {stat, rk}; }
              f = i => rank.rk[i];
            } else { const lo = Math.log10(cfg.bp_floor), hi = Math.log10(cfg.bp_max[stat] || 1); f = i => Math.max(0, (Math.log10(Math.max(b[i], cfg.bp_floor)) - lo) / (hi - lo)); }
            const per = new Float32Array(K); for (let i = 0; i < K; i++) per[i] = H * f(i);
            for (let v = 0; v < V; v++) e[v] = per[vcell[v]];
          }

          // COLOUR: FFWI at fractional hour t (linear between the two hours), viridis, per vertex.
          function paint() {
            const t0 = performance.now(), k = Math.min(F - 1, Math.floor(t)), a = Math.min(1, t - k), k1 = Math.min(F - 1, k + 1);
            const q = ffwi[stat], A = q.subarray(k * K, (k + 1) * K), B = q.subarray(k1 * K, (k1 + 1) * K);
            const hi = parseFloat(hiIn.value) * 2, c = color[cbuf ^= 1], per = new Uint8Array(K * 4);
            for (let i = 0; i < K; i++) {
              const x = A[i], y = B[i]; let v;
              if (x === 255 && y === 255) { per[4 * i + 3] = 0; continue; }
              v = x === 255 ? y : y === 255 ? x : x + (y - x) * a;
              const j = 3 * Math.min(255, Math.round(v / hi * 255));
              per[4 * i] = lut[j]; per[4 * i + 1] = lut[j + 1]; per[4 * i + 2] = lut[j + 2]; per[4 * i + 3] = 255;
            }
            for (let v = 0; v < V; v++) { const i = vcell[v] * 4, o = v * 4; c[o] = per[i]; c[o + 1] = per[i + 1]; c[o + 2] = per[i + 2]; c[o + 3] = per[i + 3]; }
            tickMs = performance.now() - t0;
            return c;
          }

          const tiles = (id, url, opacity) => new TileLayer({
            id, data: url, tileSize: 256, minZoom: 0, maxZoom: 19, opacity, pickable: false,
            renderSubLayers: p => { const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
          });
          function layers(c) {
            const out = [];
            if (useBase && cfg.base_tiles) out.push(tiles("base", cfg.base_tiles, 1.0));
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
            stamp.textContent = ts.replace("T", " ") + "Z"; slider.value = String(t);
            ruler.textContent = `${K.toLocaleString()} res ${cfg.res} cells · ${V.toLocaleString()} vertices · ${F} hours · paint ${tickMs.toFixed(0)} ms, layer ${(upMs - tickMs).toFixed(0)} ms`;
            opacV.textContent = parseFloat(opacIn.value).toFixed(2);
          }
          function legend() {
            const stops = []; for (let i = 0; i <= 8; i++) { const j = Math.round(i / 8 * 255) * 3; stops.push(`rgb(${lut[j]},${lut[j+1]},${lut[j+2]}) ${i/8*100}%`); }
            grad.style.background = `linear-gradient(90deg, ${stops.join(",")})`; hiV.textContent = parseFloat(hiIn.value).toFixed(0) + "+";
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
          function setPlaying(p) { playing = p; lastTs = 0; playBtn.textContent = p ? "pause" : "play"; playBtn.classList.toggle("hw-on", p); if (p) raf = requestAnimationFrame(frame); else cancelAnimationFrame(raf); }
          playBtn.onclick = () => setPlaying(!playing);
          q(".hw-prev").onclick = () => { setPlaying(false); t = Math.max(0, Math.ceil(t) - 1); update(); };
          q(".hw-next").onclick = () => { setPlaying(false); t = Math.min(F - 1, Math.floor(t) + 1); update(); };
          slider.oninput = () => { setPlaying(false); t = parseFloat(slider.value); update(); };
          root.querySelectorAll(".hw-st").forEach(b => b.onclick = () => { stat = b.dataset.st; root.querySelectorAll(".hw-st").forEach(x => x.classList.toggle("hw-on", x === b)); rebuildElev(); update(); if (selected >= 0) showPick(selected); });
          scaleSel.onchange = hIn.oninput = () => { rebuildElev(); update(); };
          hiIn.onchange = () => { legend(); update(); };
          opacIn.oninput = () => update();
          baseBtn.onclick = () => { useBase = !useBase; baseBtn.classList.toggle("hw-on", useBase); baseBtn.textContent = useBase ? "on" : "off"; update(); };
          q(".hw-home").onclick = () => deck.setProps({initialViewState: {...cfg.home, transitionDuration: 600}});
          q(".hw-full").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else mapEl.requestFullscreen(); };

          function boot() {
            loadStatic();
            deck = new Deck({
              parent: mapEl, controller: true, initialViewState: cfg.home, layers: [],
              getCursor: ({isHovering}) => isHovering ? "pointer" : "grab",
              onClick: info => showPick(info.index >= 0 && info.layer && info.layer.id === "hex" ? info.index : -1),
            });
            update();
            if (cfg.autoplay) setPlaying(true);
          }
          model.on("change:config", () => { loadStatic(); update(); });
          try { boot(); } catch (e) { ruler.textContent = "boot: " + e.message; console.error(e); }
          return () => { setPlaying(false); if (deck) deck.finalize(); };
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
        f"to the browser: {(hx['xy'].nbytes + hx['starts'].nbytes + 2 * hx['ffwi']['mean'].nbytes + 2 * _bpm.nbytes + 8 * K) / 1e6:.0f} MB"
    )
    return F, K, build_note, hx


@app.cell
def _(BASE_TILES, FFWI_HI, FPS, HEIGHT_M, HexWaves, RES_H, SCALE, STAT, WINDOW, colormaps, hx, json, mo, np):
    _lut = (np.asarray(colormaps["viridis"](np.linspace(0, 1, 256)))[:, :3] * 255).round().astype(int).tolist()
    _lon, _lat = hx["xy"][:, 0], hx["xy"][:, 1]
    _cfg = {
        "res": RES_H, "stat": STAT, "scale": SCALE, "height_m": HEIGHT_M, "ffwi_hi": FFWI_HI, "fps": FPS, "opacity": 0.9,
        "times": [str(t)[:16] for t in hx["times"]], "lut": _lut,
        "bp_max": {k: float(v.max()) for k, v in hx["bp"].items()}, "bp_floor": 1e-4,
        "base_tiles": BASE_TILES, "autoplay": False,
        "home": {"longitude": float((_lon.min() + _lon.max()) / 2), "latitude": float((_lat.min() + _lat.max()) / 2) - 2.0, "zoom": 3.6, "pitch": 55, "bearing": -8},
        "title": "Hex waves: burn probability as height, Fosberg fire weather as colour",
        "subtitle": f"H3 res {RES_H} · USFS FSim burn probability (270 m) x HRRR hourly FFWI (3 km), {WINDOW[0]} to {WINDOW[1]} UTC",
    }
    waves = mo.ui.anywidget(
        HexWaves(
            xy=hx["xy"].astype("<f4").tobytes(), starts=hx["starts"].astype("<u4").tobytes(), cells=hx["cells"].astype("<u8").tobytes(),
            bp_mean=hx["bp"]["mean"].astype("<f4").tobytes(), bp_max=hx["bp"]["max"].astype("<f4").tobytes(),
            ffwi_mean=hx["ffwi"]["mean"].tobytes(), ffwi_max=hx["ffwi"]["max"].tobytes(),
            config=json.dumps(_cfg),
        )
    )
    waves
    return (waves,)


@app.cell
def _(build_note, mo):
    mo.md(f"`{build_note}`")
    return


if __name__ == "__main__":
    app.run()
