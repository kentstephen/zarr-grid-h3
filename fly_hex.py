"""Headless flight of hex-waves.py: marimo run + playwright. Screenshots to shots/hex_*.png.
Usage: uv run python fly_hex.py"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent; NB = ROOT / "hex-waves.py"; SHOTS = ROOT / "shots"; SHOTS.mkdir(exist_ok=True)
PORT = 2733
srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(NB), "--headless", "--no-token", "--port", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3); logs = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1400, "height": 820})
        pg.set_default_timeout(90_000)
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}")); pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load"); t = time.perf_counter()
        pg.wait_for_selector(".hw-map > canvas", timeout=300_000)
        time.sleep(2)
        pg.screenshot(path=str(SHOTS / "hex_00_boot.png"))
        print("\n".join(logs[-25:]))
        deadline = time.time() + 120
        while time.time() < deadline and "cells" not in pg.locator(".hw-ruler").first.inner_text(): time.sleep(1)
        time.sleep(8)
        print(f"booted in {time.perf_counter()-t:.1f}s; ruler: {pg.locator('.hw-ruler').inner_text()!r}")
        pg.screenshot(path=str(SHOTS / "hex_01_home.png"))
        pg.locator(".hw-frame").fill("60"); pg.locator(".hw-frame").dispatch_event("input"); time.sleep(3)
        print("f60:", pg.locator(".hw-ruler").inner_text(), "|", pg.locator(".hw-stamp").inner_text())
        pg.screenshot(path=str(SHOTS / "hex_02_f60.png"))
        pg.click(".hw-st[data-st=max]"); time.sleep(3); print("max:", pg.locator(".hw-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "hex_03_max.png"))
        pg.select_option(".hw-scale", "linear"); time.sleep(3); pg.screenshot(path=str(SHOTS / "hex_04_linear.png"))
        pg.select_option(".hw-scale", "log")
        box = pg.locator(".hw-map").bounding_box(); pg.mouse.click(box["x"] + box["width"] * 0.38, box["y"] + box["height"] * 0.55); time.sleep(1.5)
        print("pick:", pg.locator(".hw-pick").inner_text())
        pg.screenshot(path=str(SHOTS / "hex_05_pick.png"))
        STEP = """async b => { const raf = () => new Promise(r => requestAnimationFrame(r)); await raf();
            const t = performance.now(); for (let i = 0; i < 10; i++) { b.click(); await raf(); await raf(); } return (performance.now() - t) / 10; }"""
        print(f"in-page ms per hour step: {pg.locator('.hw-next').evaluate(STEP):.0f}")
        print("\n".join(l for l in logs if "error" in l.lower() or "warn" in l.lower())[:3000])
        b.close()
finally:
    srv.terminate()
    try: out = srv.communicate(timeout=10)[0]
    except Exception: out = ""
    err = [l for l in out.splitlines() if "Traceback" in l or "Error" in l]
    if err: print("SERVER:", "\n".join(err[:20]))
