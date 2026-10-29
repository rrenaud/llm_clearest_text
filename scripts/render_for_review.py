"""Render screened answers as plain text with probe marks, for grading.

Usage: render_for_review.py screened.jsonl out.txt [eval.jsonl]
Risk spans become [[span]]{risk=.., tok=.., alts=..}; dimmed sentences become <<dim:kind>>sentence<</dim>>.
"""
import json
import re
import sys
from urllib.parse import unquote

D = lambda s: json.loads(unquote(s))


def plain(c):
    def span(m):
        d, t = D(m.group(1)), re.sub(r"</?probe-t[^>]*>", "", m.group(2))
        if "risk" not in d.get("flags", []):
            return t
        alts = ", ".join(f"{a} {p:.2f}" for a, p in d.get("alts", [])[:4])
        return f"[[{t}]]{{risk={d['risk']:.3f}, tok={d.get('tok')!r}, alts: {alts}}}"
    c = re.sub(r'<probe d="([^"]*)">([\s\S]*?)</probe>', span, c)
    c = re.sub(r'<probe-s d="([^"]*)">([\s\S]*?)</probe-s>',
               lambda m: (f"<<dim:{'+'.join(D(m.group(1))['flags'])}>>{m.group(2)}<</dim>>" if D(m.group(1)).get("flags") else m.group(2)), c)
    return re.sub(r"</?probe(?:-[st])?(?: d=\"[^\"]*\")?>", "", c)


refs = {}
if len(sys.argv) > 3:
    refs = {json.loads(l)["prompt"]: json.loads(l) for l in open(sys.argv[3])}
with open(sys.argv[2], "w") as f:
    for i, l in enumerate(open(sys.argv[1])):
        r = json.loads(l)
        f.write(f"=== #{i} QUERY: {r['query']}\n")
        if r["query"] in refs:
            e = refs[r["query"]]
            f.write(f"REFERENCE: {e['reference']}\nGRADING NOTES: {e['grading_notes']}\n")
        f.write(f"--- ANSWER\n{plain(r['content'])}\n\n")
