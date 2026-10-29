"""Screenshot saved Open WebUI chats with the sidebar hidden, optionally hovering a probe span.

Usage: python scripts/screenshot_chat.py CHAT_ID out.png [--hover N] [--scroll-to TEXT] [--webui URL]
"""
import argparse

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("chat_id")
ap.add_argument("out")
ap.add_argument("--hover", type=int, help="hover the Nth flagged (.probe-risk) span")
ap.add_argument("--hover-dim", type=int, help="hover the Nth dimmed sentence")
ap.add_argument("--scroll-to", help="scroll so this text is near the top")
ap.add_argument("--webui", default="http://127.0.0.1:8502")
ap.add_argument("--height", type=int, default=820)
args = ap.parse_args()

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1000, "height": args.height}, device_scale_factor=2, color_scheme="light")
    pg.goto(f"{args.webui}/c/{args.chat_id}")
    pg.wait_for_selector("#response-content-container", timeout=60000)
    pg.wait_for_timeout(1500)
    # Hide the sidebar (chat titles) and anything outside the conversation.
    pg.add_style_tag(content="#sidebar, nav#sidebar, [id^=sidebar], [data-sonner-toaster], .toaster {display:none !important}")
    for sel in ["button[aria-label='Close Sidebar']", "#sidebar-toggle-button"]:
        if pg.query_selector(sel):
            pg.click(sel)
    pg.wait_for_timeout(500)
    if args.scroll_to:
        pg.get_by_text(args.scroll_to).first.evaluate("e => e.scrollIntoView({block: 'center'})")
    target = None
    if args.hover is not None:
        target = pg.query_selector_all(".probe-risk")[args.hover]
    elif args.hover_dim is not None:
        target = pg.query_selector_all(".probe-dim")[args.hover_dim]
    if target:
        target.scroll_into_view_if_needed()
        target.evaluate("e => e.scrollIntoView({block: 'center'})")
        target.hover()
        pg.wait_for_timeout(800)
    pg.screenshot(path=args.out)
    b.close()
