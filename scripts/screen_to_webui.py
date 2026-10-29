"""Run queries through the probed server and import each answer (with probe tags) as an Open WebUI chat.

Usage: screen_to_webui.py queries.txt out.jsonl [--model M] [--folder unreviewed] [--webui http://127.0.0.1:8502]
       [--temperature 0] [--top-p 1]  (Qwen3.6 without thinking loops at temperature 0; use 0.7 / 0.8)
Move chats between folders later with --move out.jsonl '{"query": "folder", ...}'.
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def call(url, data=None, token=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, json.dumps(data).encode() if data is not None else None, headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def webui_token(webui):
    # WEBUI_AUTH=False signs everyone in as the built-in admin.
    return call(f"{webui}/api/v1/auths/signin", {"email": "", "password": ""})["token"]


def folder_id(webui, token, name, cache={}):
    if name not in cache:
        existing = {f["name"]: f["id"] for f in call(f"{webui}/api/v1/folders/", token=token)}
        cache[name] = existing.get(name) or call(f"{webui}/api/v1/folders/", {"name": name}, token)["id"]
    return cache[name]


def generate(model, query, temperature=0.0, top_p=1.0):
    url = (ROOT / ".modal_url").read_text().strip()
    key = (ROOT / ".modal_api_key").read_text().strip()
    body = {"model": model, "messages": [{"role": "user", "content": query}], "temperature": temperature,
            "top_p": top_p, "stream": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    content = ""
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if line.startswith("data: {"):
            content += json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
    return content


def chat_json(model, query, content):
    now = int(time.time())
    u, a = str(uuid.uuid4()), str(uuid.uuid4())
    user = {"id": u, "parentId": None, "childrenIds": [a], "role": "user", "content": query, "timestamp": now, "models": [model]}
    asst = {"id": a, "parentId": u, "childrenIds": [], "role": "assistant", "content": content, "model": model,
            "modelName": model, "modelIdx": 0, "done": True, "timestamp": now}
    return {"title": query[:80], "models": [model], "messages": [user, asst],
            "history": {"messages": {u: user, a: asst}, "currentId": a}, "timestamp": now * 1000}


def flags(content):
    risk = [t for d, t in re.findall(r'<probe d="([^"]*)">(.*?)</probe>', content)
            if "risk" in json.loads(urllib.parse.unquote(d)).get("flags", [])]
    dim = sum(1 for _ in re.finditer(r"<probe-s ", content))
    return risk, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("out")
    ap.add_argument("--model", default="qwen2.5-7b-instruct-probed")
    ap.add_argument("--folder", default="unreviewed")
    ap.add_argument("--webui", default="http://127.0.0.1:8502")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--move", help="JSON mapping query -> folder; moves chats recorded in OUT")
    args = ap.parse_args()
    token = webui_token(args.webui)

    if args.move:
        mapping = json.loads(args.move)
        for line in open(args.out):
            r = json.loads(line)
            if r["query"] in mapping:
                fid = folder_id(args.webui, token, mapping[r["query"]])
                call(f"{args.webui}/api/v1/chats/{r['chat_id']}/folder", {"folder_id": fid}, token)
                print(f"{mapping[r['query']]:>10} <- {r['query']}")
        return

    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["query"] for l in open(args.out)}
    fid = folder_id(args.webui, token, args.folder)
    for q in [l.strip() for l in open(args.queries) if l.strip()]:
        if q in done:
            continue
        content = generate(args.model, q, args.temperature, args.top_p)
        chat = call(f"{args.webui}/api/v1/chats/new", {"chat": chat_json(args.model, q, content), "folder_id": fid}, token)
        risk, dim = flags(content)
        with open(args.out, "a") as f:
            f.write(json.dumps({"query": q, "chat_id": chat["id"], "risk": risk, "dim": dim, "content": content}) + "\n")
        print(f"{len(risk)} risk {dim} dim | {q} | {risk}", flush=True)


if __name__ == "__main__":
    main()
