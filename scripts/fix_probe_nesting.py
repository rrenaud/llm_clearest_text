"""Repair chats saved by older servers: cross-nested tags ('</probe-s></probe>') and tags inside LaTeX math."""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
from screen_to_webui import ROOT, call, webui_token  # noqa: E402
from segment import MATH  # noqa: E402

TAG = re.compile(r'</?probe(?:-[st])?(?: d="[^"]*")?>')


def repair(content):
    content = content.replace("</probe-s></probe>", "</probe></probe-s>")
    return MATH.sub(lambda m: TAG.sub("", m.group(0)), content)


webui = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8502"
db = sqlite3.connect(f"file:{ROOT}/webui-data/webui.db?mode=ro", uri=True)
ids = [r[0] for r in db.execute("select id from chat where chat like '%<probe%'")]
token = webui_token(webui)
for cid in ids:
    chat = call(f"{webui}/api/v1/chats/{cid}", token=token)["chat"]
    changed = False
    for m in list(chat.get("messages", [])) + list(chat.get("history", {}).get("messages", {}).values()):
        if isinstance(m.get("content"), str) and repair(m["content"]) != m["content"]:
            m["content"] = repair(m["content"])
            changed = True
    if changed:
        call(f"{webui}/api/v1/chats/{cid}", {"chat": chat}, token)
        print("fixed", cid, chat.get("title", "")[:60])
