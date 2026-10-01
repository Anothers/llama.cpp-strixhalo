import json, sys, urllib.request
S = "/home/zhangde/bench/_scratch/"
real = open(S + "real_text.txt", encoding="utf-8", errors="replace").read()
PROMPTS = [
    "用 Python 写一个线程安全的 LRU 缓存类，带类型注解和 docstring。",
    "解释 TCP 三次握手和四次挥手，分点说明。",
    "写一个快速排序的 C++ 实现，并分析时间复杂度。",
    "比较 Rust 和 Go 在并发模型上的差异，给出代码示例。",
    "列出 10 个常用的 Linux 网络排查命令并说明用途。",
    real[:96000] + "\n\n请用三句话总结上文的主要内容。",
]
cfg = sys.argv[1]

def chat(p, lp):
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if lp:
        body.update({"logprobs": True, "top_logprobs": 5})
    r = urllib.request.Request("http://127.0.0.1:8094/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))

if cfg != "compare":
    out = []
    for p in PROMPTS:
        d = chat(p, cfg == "A")
        c = d["choices"][0]; t = d["timings"]
        rec = {"text": c["message"]["content"], "acc": t.get("draft_n_accepted"), "dn": t.get("draft_n")}
        if cfg == "A":
            rec["lp"] = [[e["token"], e["logprob"], [[x["token"], x["logprob"]] for x in e["top_logprobs"]]]
                         for e in c["logprobs"]["content"]]
        out.append(rec)
        print(f"  [{cfg}] {p[:14]!r:24} {len(rec['text'])} chars  accept {rec['acc']}/{rec['dn']}", flush=True)
    json.dump(out, open(S + f"speccorr_{cfg}.json", "w"), ensure_ascii=False)
    sys.exit()

A = json.load(open(S + "speccorr_A.json"))
for cfg in [c for c in ("B", "B0") if __import__("os").path.exists(S + f"speccorr_{c}.json") and __import__("os").path.getmtime(S + f"speccorr_{c}.json") >= __import__("os").path.getmtime(S + "speccorr_A.json")]:
    X = json.load(open(S + f"speccorr_{cfg}.json"))
    print(f"== {cfg} vs A (no MTP)")
    for i, (a, x) in enumerate(zip(A, X)):
        if a["text"] == x["text"]:
            print(f"  prompt {i}: identical ({len(a['text'])} chars), accept {x['acc']}/{x['dn']}")
            continue
        k = 0
        while k < min(len(a["text"]), len(x["text"])) and a["text"][k] == x["text"][k]:
            k += 1
        # token of A that contains char k
        pos = 0; ti = None
        for j, (tok, lp, top) in enumerate(a["lp"]):
            if pos + len(tok) > k:
                ti = j; break
            pos += len(tok)
        tok, lp, top = a["lp"][ti]
        alt = x["text"][pos:pos + 12]
        alt_lp = None
        for t2, l2 in top:
            if t2 and alt.startswith(t2) and t2 != tok:
                alt_lp = l2; break
        gap = (lp - alt_lp) if alt_lp is not None else None
        gs = f"{gap:.3f}" if gap is not None else "not in A top-5"
        rank = next((r for r, (t2, l2) in enumerate(top) if t2 and alt.startswith(t2) and t2 != tok), None)
        verdict = "NEAR-TIE" if (gap is not None and gap < 0.15 and rank is not None and rank <= 1) else "BAD"
        print(f"  prompt {i}: {verdict} diverges at char {k} (A token {ti}): A {tok!r} {lp:.3f} | {cfg} {alt!r} | gap {gs} | A top: "
              + ", ".join(f"{t}:{l:.2f}" for t, l in top[:4]) + f"  accept {x['acc']}/{x['dn']}")
