"""Headless flight of hrrr-heat-domes.py: marimo run + playwright. Screenshots to
shots/, console to stdout. Usage: uv run python fly.py [notebook.py] [shots_dir]"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent
NB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "hrrr-heat-domes.py"
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "shots"
SHOTS.mkdir(exist_ok=True)
PORT = 2732
srv = subprocess.Popen([sys.executable, "-m", "marimo", "run", str(NB), "--headless", "--no-token", "--port", str(PORT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(3)
logs = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1400, "height": 820})
        pg.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
        t = time.perf_counter()
        pg.wait_for_selector(".rf-map > canvas", timeout=240_000)
        # the geometry lands first (ruler says "0 frames"), the frames after the read
        deadline = time.time() + 300
        while time.time() < deadline:
            txt = pg.locator(".rf-ruler").inner_text()
            if "frames" in txt and " 0 frames" not in txt:
                break
            time.sleep(2)
        else:
            print("WAIT FAILED; ruler:", pg.locator(".rf-ruler").inner_text())
            pg.screenshot(path=str(SHOTS / "00-fail.png")); print("\n".join(logs[-40:])); raise SystemExit(1)
        time.sleep(6)  # tiles + first texture
        print(f"booted in {time.perf_counter()-t:.1f}s; ruler: {pg.locator('.rf-ruler').inner_text()!r}")
        print("note:", pg.locator(".rf-note").inner_text())
        pg.screenshot(path=str(SHOTS / "01-index.png"))
        box = pg.locator(".rf-map").bounding_box()
        pg.mouse.click(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5)
        time.sleep(1.5)
        print("pick:", pg.locator(".rf-cname").inner_text(), "|", pg.locator(".rf-how").inner_text(), "|", pg.locator(".rf-cval").inner_text())
        pg.screenshot(path=str(SHOTS / "02-pick.png"))
        pg.click(".rf-fi[data-field=load]")
        pg.locator(".rf-frame").fill("88"); pg.locator(".rf-frame").dispatch_event("input")
        time.sleep(2.5)
        print("ruler load f88:", pg.locator(".rf-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "03-load-f88.png"))
        pg.select_option(".rf-rule", "pixel"); time.sleep(2)
        print("ruler pixel rule:", pg.locator(".rf-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "04-load-pixelrule.png"))
        pg.select_option(".rf-rule", "all"); time.sleep(2)
        print("ruler all rule:", pg.locator(".rf-ruler").inner_text())
        # frame stepping speed: 20 steps
        t2 = time.perf_counter()
        for _ in range(20):
            pg.click(".rf-next")
        print(f"20 steps in {time.perf_counter()-t2:.1f}s")
        time.sleep(2); print("ruler after steps:", pg.locator(".rf-ruler").inner_text())
        STEP = """async b => { const raf = () => new Promise(r => requestAnimationFrame(r)); await raf();
            const t = performance.now(); for (let i = 0; i < 10; i++) { b.click(); await raf(); await raf(); } return (performance.now() - t) / 10; }"""
        print(f"in-page ms per step, boundaries on (all rule): {pg.locator('.rf-next').evaluate(STEP):.0f}")
        pg.select_option(".rf-rule", "majority"); time.sleep(1)
        print(f"in-page ms per step, boundaries on (majority): {pg.locator('.rf-next').evaluate(STEP):.0f}")
        pg.click(".rf-bnd"); time.sleep(1)
        print(f"in-page ms per step, boundaries off: {pg.locator('.rf-next').evaluate(STEP):.0f}")
        pg.click(".rf-fi[data-field=index]"); time.sleep(1)
        print(f"in-page ms per step, heat index, boundaries off: {pg.locator('.rf-next').evaluate(STEP):.0f}")
        print("ruler:", pg.locator(".rf-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "05-steps.png"))
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    print("--- console ---"); print("\n".join(l for l in logs if "deck" in l.lower() or "error" in l.lower() or "warn" in l.lower())[:4000])
    print("--- server tail ---"); print(out[-2500:])
