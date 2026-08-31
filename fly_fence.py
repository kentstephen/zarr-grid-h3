"""Headless flight of storm-fence.py: marimo run + playwright. Screenshots to shots/,
console to stdout. Usage: uv run python fly_fence.py [notebook.py] [shots_dir]"""
import pathlib, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).parent
NB = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "storm-fence.py"
SHOTS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "shots"
SHOTS.mkdir(exist_ok=True)
PORT = 2733
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
        pg.wait_for_selector(".sf-map canvas", timeout=240_000)
        deadline = time.time() + 1200
        while time.time() < deadline:
            txt = pg.locator(".sf-ruler").inner_text()
            if "frames" in txt and " 0 frames" not in txt:
                break
            time.sleep(2)
        else:
            print("WAIT FAILED; ruler:", pg.locator(".sf-ruler").inner_text())
            pg.screenshot(path=str(SHOTS / "sf-00-fail.png")); print("\n".join(logs[-40:])); raise SystemExit(1)
        time.sleep(6)  # tiles + first texture
        print(f"booted in {time.perf_counter()-t:.1f}s; ruler: {pg.locator('.sf-ruler').inner_text()!r}")
        print("note:", pg.locator(".sf-note").inner_text())
        print("fnote:", pg.locator(".sf-fnote").inner_text())
        pg.screenshot(path=str(SHOTS / "sf-01-field.png"))
        # pick: click mid-map for the two-line chart
        box = pg.locator(".sf-map").bounding_box()
        pg.mouse.click(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.5)
        time.sleep(1.5)
        print("pick:", pg.locator(".sf-cname").inner_text(), "|", pg.locator(".sf-how.sf-num").inner_text(), "|", pg.locator(".sf-cval").inner_text())
        pg.screenshot(path=str(SHOTS / "sf-02-pick.png"))
        # threshold up: the fence should retreat to cores
        pg.locator(".sf-thr").fill("4"); pg.locator(".sf-thr").dispatch_event("input")
        time.sleep(1.5)
        print("ruler thr4:", pg.locator(".sf-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "sf-03-thr4.png"))
        pg.locator(".sf-thr").fill("1"); pg.locator(".sf-thr").dispatch_event("input"); time.sleep(1)
        for rule in ("any", "all", "majority"):
            pg.select_option(".sf-rule", rule); time.sleep(1.2)
            print(f"ruler {rule}:", pg.locator(".sf-ruler").inner_text().replace("\n", " | "))
        if pg.locator(".sf-csi").count():
            time.sleep(1.5)
            print("match:", pg.locator(".sf-csi").inner_text(), "|", pg.locator(".sf-podfar").inner_text(), "|", pg.locator(".sf-hmf").inner_text())
        pg.screenshot(path=str(SHOTS / "sf-04-majority.png"))
        # frame stepping cost, fence on and off
        STEP = """async b => { const raf = () => new Promise(r => requestAnimationFrame(r)); await raf();
            const t = performance.now(); for (let i = 0; i < 10; i++) { b.click(); await raf(); await raf(); } return (performance.now() - t) / 10; }"""
        print(f"in-page ms per step, fence on: {pg.locator('.sf-next').evaluate(STEP):.0f}")
        pg.click(".sf-fn"); time.sleep(1)
        print(f"in-page ms per step, fence off: {pg.locator('.sf-next').evaluate(STEP):.0f}")
        pg.click(".sf-fn"); time.sleep(1)
        print("ruler end:", pg.locator(".sf-ruler").inner_text())
        pg.screenshot(path=str(SHOTS / "sf-05-end.png"))
        b.close()
finally:
    srv.terminate()
    out = srv.stdout.read() if srv.stdout else ""
    print("--- console ---"); print("\n".join(l for l in logs if "deck" in l.lower() or "error" in l.lower() or "warn" in l.lower())[:4000])
    print("--- server tail ---"); print(out[-2500:])
