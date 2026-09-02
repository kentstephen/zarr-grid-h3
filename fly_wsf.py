"""Headless flight of s2-wsf-aef-overture-pair.py: marimo run + playwright. Screenshots to
shots/wsf/, console to stdout. Usage: uv run python fly_wsf.py [notebook.py] [shots_dir]"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent
NB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "s2-wsf-aef-overture-pair.py"
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "shots" / "wsf"
SHOTS.mkdir(parents=True, exist_ok=True)
PORT = 2735
srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(NB), "--headless", "--no-token", "--port", str(PORT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3)
logs = []


def status(pg):
    try:
        return pg.locator(".sp-status").inner_text()
    except Exception as e:
        return f"<no status: {e}>"


def wait_status(pg, pred, secs, what):
    deadline = time.time() + secs
    while time.time() < deadline:
        s = status(pg)
        if pred(s):
            return s
        time.sleep(1)
    print(f"WAIT FAILED ({what}); status: {status(pg)!r}")
    pg.screenshot(path=str(SHOTS / "00-fail.png"))
    print("\n".join(logs[-40:]))
    raise SystemExit(1)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 900})
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
        t = time.perf_counter()
        pg.wait_for_selector(".sp-right canvas", timeout=240_000)
        s = wait_status(pg, lambda s: "cells" in s or "moved" in s or "failed" in s, 400, "first fold")
        time.sleep(8)  # S2 tiles
        print(f"booted in {time.perf_counter() - t:.1f}s")
        print("status:", status(pg))
        print("layers:", pg.evaluate("window.__spLayers()"))
        print("tiles:", pg.evaluate("window.__spTiles"))
        pg.screenshot(path=str(SHOTS / "01-home.png"), full_page=True)
        # the widget sits below the fold (the intro is long): bring it into the
        # viewport before the pointer work, or the hover lands off-screen
        pg.locator(".sp-root").scroll_into_view_if_needed()
        time.sleep(1)
        # hover the right pane's middle: the ring should mirror on the left
        box = pg.locator(".sp-right").bounding_box()
        pg.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        time.sleep(1)
        print("hover layers:", pg.evaluate("window.__spLayers()"))
        print("place under pointer:", pg.locator(".sp-place").inner_text()[:200])
        pg.screenshot(path=str(SHOTS / "02-hover.png"))
        pg.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        time.sleep(2)
        print("panel:", pg.locator(".sp-panel").inner_text()[:300])
        pg.screenshot(path=str(SHOTS / "03-pick.png"))
        # the fills
        for k, nm in ((2, "grew"), (3, "byear"), (4, "shift"), (5, "when")):
            pg.click(f".sp-fill[data-value={nm}]")
            time.sleep(2)
            print(f"fill {nm}:", status(pg)[-120:])
            print("legend:", pg.locator(".sp-legend").inner_text()[:300])
            pg.screenshot(path=str(SHOTS / f"04-fill-{nm}.png"), full_page=True)
        # the built fill: the raster on the right, no hexagons
        pg.click(".sp-fill[data-value=built]")
        time.sleep(4)
        print("fill built layers:", pg.evaluate("window.__spLayers()"), "legend:", pg.locator(".sp-legend").inner_text()[:80].replace("\n", " "))
        pg.screenshot(path=str(SHOTS / "04-fill-built.png"))
        pg.click(".sp-fill[data-value=grew]")
        time.sleep(2)
        # res 12: zoom 14.8 over Paradise
        t3 = time.perf_counter()
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-121.62, 39.76], zoom: 14.8})")
        s = wait_status(pg, lambda s: "res 12" in s and ("cells" in s or "grew" in s), 300, "res 12")
        time.sleep(6)
        print(f"res 12 in {time.perf_counter() - t3:.1f}s:", status(pg)[:400])
        print("res 12 layers:", pg.evaluate("window.__spLayers()"))
        pg.screenshot(path=str(SHOTS / "10-res12.png"))
        pg.click(".sp-fill[data-value=built]")
        time.sleep(6)
        print("res 12 built tiles:", pg.evaluate("window.__spTiles"))
        pg.screenshot(path=str(SHOTS / "11-res12-built.png"))
        pg.click(".sp-fill[data-value=grew]")
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-121.6, 39.76], zoom: 10.2})")
        s = wait_status(pg, lambda s: "res 8" in s and "folding" not in s, 300, "back to res 8")
        time.sleep(2)
        # the S2 year: 2023 (a tile layer swap, no fold)
        pg.click(".sp-s2y[data-value='2023']")
        time.sleep(10)
        print("s2 2023 tiles:", pg.evaluate("window.__spTiles"))
        pg.screenshot(path=str(SHOTS / "05-s2-2023.png"))
        # the window: to end one year later (a frame rebuild + one more AEF year, no WSF refold)
        pg.locator(".sp-root").focus()
        pg.keyboard.press("+")
        s = wait_status(pg, lambda s: "2024" in s and ("cells" in s or "moved" in s or "grew" in s), 300, "window to 2024")
        time.sleep(3)
        print("window to 2024:", status(pg))
        pg.screenshot(path=str(SHOTS / "06-window-2024.png"))
        # camera sync: drag the left map, both should move
        pg.locator(".sp-root").scroll_into_view_if_needed()
        time.sleep(1)
        lb = pg.locator(".sp-left").bounding_box()
        before = pg.evaluate("window.__spMaps().map(m => [m.getCenter().lng, m.getCenter().lat, m.getZoom()])")
        pg.mouse.move(lb["x"] + lb["width"] * 0.5, lb["y"] + lb["height"] * 0.5)
        pg.mouse.down()
        pg.mouse.move(lb["x"] + lb["width"] * 0.3, lb["y"] + lb["height"] * 0.4, steps=10)
        pg.mouse.up()
        time.sleep(1)
        after = pg.evaluate("window.__spMaps().map(m => [m.getCenter().lng, m.getCenter().lat, m.getZoom()])")
        print("cameras before:", before)
        print("cameras after: ", after)
        s = wait_status(pg, lambda s: "folding" not in s, 300, "refold after drag")
        time.sleep(4)
        print("after drag:", status(pg))
        pg.screenshot(path=str(SHOTS / "07-dragged.png"))
        # perimeters present on both maps?
        print("fullscreen ctrl on right map:", pg.locator(".sp-right .maplibregl-ctrl-fullscreen").count(),
              "| strip bg/color:", pg.locator(".sp-status").evaluate("e => [getComputedStyle(e.parentElement).backgroundColor, getComputedStyle(e).color]"))
        print("bounds:", pg.evaluate("window.__spMaps().map(m => [!!m.getSource('div'), !!m.getLayer('div-region'), !!m.getLayer('div-county'), !!m.getLayer('div-locality')])"))
        print("division features rendered (region, county, locality):", pg.evaluate("window.__spMaps().map(m => ['div-region', 'div-county', 'div-locality'].map(l => m.queryRenderedFeatures({layers: [l]}).length))"))
        print("layer order R:", pg.evaluate("window.__spMaps()[1].getStyle().layers.map(l => l.id).filter(id => /div-|s2-|wsf-|hexes|hover|picked|watername_ocean/.test(id))"))
        # zoom out below the hexagon zoom: WSF pyramid right, S2 from L5 left
        t3 = time.perf_counter()
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-121.6, 39.76], zoom: 8.0})")
        s = wait_status(pg, lambda s: "raster on the right" in s, 120, "zoom 8 status")
        for _ in range(60):
            tl = pg.evaluate("window.__spTiles")
            if tl["asked"] - tl["got"] - tl["empty"] - tl["err"] - tl["abort"] == 0 and time.perf_counter() - t3 > 4:
                break
            time.sleep(1)
        print(f"zoom 8 tiles settled in {time.perf_counter() - t3:.1f}s:", pg.evaluate("window.__spTiles"))
        print("zoom 8 layers:", pg.evaluate("window.__spLayers()"))
        print("zoom 8 divisions (region, county, locality):", pg.evaluate("window.__spMaps().map(m => ['div-region', 'div-county', 'div-locality'].map(l => m.queryRenderedFeatures({layers: [l]}).length))"))
        pg.screenshot(path=str(SHOTS / "08-zoom8.png"))
        t3 = time.perf_counter()
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-121.6, 39.76], zoom: 7.0})")
        for _ in range(90):
            tl = pg.evaluate("window.__spTiles")
            if tl["asked"] - tl["got"] - tl["empty"] - tl["err"] - tl["abort"] == 0 and time.perf_counter() - t3 > 4:
                break
            time.sleep(1)
        print(f"zoom 7 tiles settled in {time.perf_counter() - t3:.1f}s:", pg.evaluate("window.__spTiles"))
        print("zoom 7 divisions (region, county, locality):", pg.evaluate("window.__spMaps().map(m => ['div-region', 'div-county', 'div-locality'].map(l => m.queryRenderedFeatures({layers: [l]}).length))"))
        pg.screenshot(path=str(SHOTS / "09-zoom7.png"))
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    print("--- console ---")
    print("\n".join(l for l in logs if "deck" in l.lower() or "error" in l.lower() or "warn" in l.lower())[:4000])
    print("--- server tail ---")
    print(out[-3000:])
