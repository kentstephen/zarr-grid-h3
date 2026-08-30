"""Headless flight: marimo run + playwright. Screenshots to proto/shots/, console to stdout."""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent; SHOTS = ROOT / "shots"; SHOTS.mkdir(exist_ok=True)
PORT = 2731
srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(ROOT / "raster_mesh.py"), "--headless", "--no-token", "--port", str(PORT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3)
logs = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1400, "height": 800})
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
        t = time.perf_counter()
        pg.wait_for_selector(".rf-map canvas", timeout=180_000)
        try:
            pg.locator(".rf-ruler").filter(has_text="quads").wait_for(timeout=150_000)
        except Exception as e:
            print("WAIT FAILED:", str(e).splitlines()[0]); print("ruler:", pg.text_content(".rf-ruler")); print("pick:", pg.text_content(".rf-pick"))
            pg.screenshot(path=str(SHOTS / "00-fail.png")); print("\n".join(logs[-40:])); raise SystemExit(1)
        time.sleep(8)  # tiles + first texture
        print(f"booted in {time.perf_counter()-t:.1f}s; ruler: {pg.text_content('.rf-ruler')!r}")
        pg.screenshot(path=str(SHOTS / "01-index.png"))
        box = pg.locator(".rf-map").bounding_box()
        pg.mouse.click(box["x"] + box["width"] * 0.52, box["y"] + box["height"] * 0.55)
        time.sleep(1.5)
        print("pick:", pg.inner_text(".rf-pick"))
        pg.screenshot(path=str(SHOTS / "02-pick.png"))
        pg.click(".rf-fi[data-field=load]")
        pg.locator(".rf-frame").fill("40"); pg.locator(".rf-frame").dispatch_event("input")
        time.sleep(2)
        print("ruler:", pg.text_content(".rf-ruler"))
        pg.screenshot(path=str(SHOTS / "03-load-f40.png"))
        pg.select_option(".rf-rule", "pixel"); time.sleep(1.5)
        print("ruler pixel rule:", pg.text_content(".rf-ruler"))
        pg.screenshot(path=str(SHOTS / "04-load-pixelrule.png"))
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
print("--- console ---"); print("\n".join(l for l in logs if "deck" in l.lower() or "error" in l.lower() or "warn" in l.lower())[:4000])
print("--- server tail ---"); print(out[-1500:])
