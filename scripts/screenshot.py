"""Drive the Open WebUI fork headlessly: send a prompt, wait for the answer, save screenshots.

Usage: python scripts/screenshot.py "prompt" out_prefix [model]
"""
import sys

from playwright.sync_api import sync_playwright

prompt, out = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else None
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
    pg.goto("http://127.0.0.1:8080/" + (f"?model={model}" if model else ""))
    pg.wait_for_selector("#chat-input", timeout=60000)
    pg.keyboard.press("Escape")
    pg.click("#chat-input")
    pg.keyboard.type(prompt)
    pg.keyboard.press("Enter")
    # Wait until the response text is non-empty and stable for 6 s.
    pg.wait_for_selector("#response-content-container", timeout=120000)
    last, stable = None, 0
    while stable < 6:
        pg.wait_for_timeout(1000)
        cur = pg.inner_text("#response-content-container")
        stable = stable + 1 if cur == last and cur.strip() else 0
        last = cur
    pg.screenshot(path=f"{out}.png", full_page=True)
    risky = pg.query_selector(".probe-risk") or pg.query_selector(".probe-span")
    if risky:
        risky.hover()
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{out}_hover.png")
    print("probe spans:", len(pg.query_selector_all(".probe-span")), "risk:", len(pg.query_selector_all(".probe-risk")),
          "dim:", len(pg.query_selector_all(".probe-dim")))
    print(last[:1500])
    b.close()
