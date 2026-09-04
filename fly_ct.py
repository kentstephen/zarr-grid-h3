"""Headless flight of s2-ctrees-pair.py: marimo run + playwright. Screenshots to
shots/ct/, console to stdout. Usage: uv run python fly_ct.py [notebook.py] [shots_dir]"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent
NB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "s2-ctrees-pair.py"
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "shots" / "ct"
SHOTS.mkdir(parents=True, exist_ok=True)
PORT = 2736
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


def click_mid(pg, sel):
    box = pg.locator(sel).bounding_box()
    x, y = box["x"] + box["width"] * 0.5, min(box["y"] + box["height"] * 0.5, 850)
    pg.mouse.move(x, y)
    time.sleep(0.8)
    pg.mouse.click(x, y)


try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 900})
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
        t = time.perf_counter()
        pg.wait_for_selector(".sp-right canvas", timeout=240_000)
        s = wait_status(pg, lambda s: "cells" in s or "failed" in s, 400, "first fold")
        time.sleep(8)
        print(f"booted in {time.perf_counter() - t:.1f}s")
        print("status:", status(pg))
        print("layers:", pg.evaluate("window.__spLayers()"))
        print("tiles:", pg.evaluate("window.__spTiles"))
        print("legend:", pg.locator(".sp-legend").inner_text()[:200])
        pg.screenshot(path=str(SHOTS / "01-home.png"), full_page=True)
        pg.locator(".sp-root").scroll_into_view_if_needed()
        time.sleep(1)
        # hover + click the right pane's middle
        click_mid(pg, ".sp-right")
        time.sleep(2)
        print("panel:", pg.locator(".sp-panel").inner_text()[:400])
        print("panel svg:", pg.locator(".sp-panel svg").count())
        pg.screenshot(path=str(SHOTS / "02-pick.png"))
        for nm in ("lossyear", "stock", "change"):
            pg.click(f".sp-fill[data-value={nm}]")
            time.sleep(2)
            print(f"fill {nm}:", status(pg)[-160:])
            print("legend:", pg.locator(".sp-legend").inner_text()[:200])
            pg.screenshot(path=str(SHOTS / f"03-fill-{nm}.png"), full_page=True)
        # the window: from end forward (a frame rebuild, no fetch)
        pg.locator(".sp-root").focus()
        t3 = time.perf_counter()
        pg.keyboard.press("=")
        s = wait_status(pg, lambda s: "2001 to 2025" in s and "cells" in s, 120, "window from 2001")
        print(f"window 2001..2025 in {time.perf_counter() - t3:.1f}s:", status(pg)[:200])
        pg.keyboard.press("_")
        s = wait_status(pg, lambda s: "2001 to 2024" in s and "cells" in s, 120, "window to 2024")
        print("window 2001..2024:", status(pg)[:200])
        pg.screenshot(path=str(SHOTS / "04-window.png"))
        # the picture slider: 2023 (S2) then 2009 (Landsat, not fetched)
        pg.keyboard.press("]")
        time.sleep(6)
        print("picture 2023:", pg.locator(".sp-yeartxt").inner_text(), pg.evaluate("window.__spLayers()['left']"))
        for _ in range(15):
            pg.keyboard.press("[")
        time.sleep(1)
        print("picture after 15 steps back:", pg.locator(".sp-yeartxt").inner_text(), pg.evaluate("window.__spLayers()['left']"))
        pg.screenshot(path=str(SHOTS / "05-picture-ls.png"))
        # NDVI mode: the S2 layer id changes, tiles re-asked
        pg.keyboard.press("n")
        time.sleep(8)
        print("ndvi mode:", pg.evaluate("window.__spLayers()['left']"), pg.evaluate("window.__spTiles"))
        pg.screenshot(path=str(SHOTS / "05b-ndvi.png"))
        pg.keyboard.press("n")
        time.sleep(2)
        # ask for the Landsat still (no key here: the status says so; with a key it fetches)
        pg.keyboard.press("k")
        time.sleep(3)
        print("landsat k:", status(pg)[:200])
        pg.click(".sp-ls[data-value=all]")
        time.sleep(3)
        print("landsat all:", status(pg)[:200])
        # camera sync: drag the left map
        lb = pg.locator(".sp-left").bounding_box()
        before = pg.evaluate("window.__spMaps().map(m => [m.getCenter().lng, m.getCenter().lat, m.getZoom()])")
        pg.mouse.move(lb["x"] + lb["width"] * 0.5, min(lb["y"] + lb["height"] * 0.5, 850))
        pg.mouse.down()
        pg.mouse.move(lb["x"] + lb["width"] * 0.3, min(lb["y"] + lb["height"] * 0.4, 800), steps=10)
        pg.mouse.up()
        time.sleep(1)
        after = pg.evaluate("window.__spMaps().map(m => [m.getCenter().lng, m.getCenter().lat, m.getZoom()])")
        print("cameras before:", before)
        print("cameras after: ", after)
        s = wait_status(pg, lambda s: "folding" not in s, 300, "refold after drag")
        time.sleep(3)
        print("after drag:", status(pg)[:300])
        pg.screenshot(path=str(SHOTS / "06-dragged.png"))
        # zoom 12: still res 9
        t3 = time.perf_counter()
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-61.95, -10.88], zoom: 12.5})")
        s = wait_status(pg, lambda s: "res 9" in s and "folding" not in s, 300, "zoom 12.5")
        time.sleep(5)
        print(f"zoom 12.5 in {time.perf_counter() - t3:.1f}s:", status(pg)[:300])
        pg.screenshot(path=str(SHOTS / "07-zoom12.png"))
        # zoom 8: hexes off
        pg.evaluate("window.__spMaps()[0].jumpTo({center: [-61.95, -10.88], zoom: 8.0})")
        s = wait_status(pg, lambda s: "zoom in past" in s, 120, "zoom 8 status")
        time.sleep(4)
        print("zoom 8:", status(pg)[:200], pg.evaluate("window.__spLayers()"))
        pg.screenshot(path=str(SHOTS / "08-zoom8.png"))
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    print("--- console ---")
    print("\n".join(l for l in logs if "deck" in l.lower() or "error" in l.lower() or "warn" in l.lower())[:4000])
    print("--- server tail ---")
    print(out[-3000:])
