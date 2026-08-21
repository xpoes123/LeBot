"""LeBot live answerer — a tiny local web app. No bot, no buzzing.

You're in a Discord call hearing the reader. Press ▶ Start when a question begins and
■ End when it finishes; between them the app captures your computer audio and streams it
to Deepgram for clean transcription. On End it sends the complete question to LeBot's
full-accuracy path and shows a confident answer with an explanation.

Run:  ./botvenv/bin/python live.py     then open http://localhost:7788
Env:  DEEPGRAM_API_KEY (required)   AUDIO_SOURCE (default: default sink monitor)
      LEBOT_URL (default VPS)
Audio only leaves your machine to Deepgram while a question is running.
"""
import asyncio
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import websockets

from listen import _default_monitor

DG_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
LEBOT_URL = os.environ.get("LEBOT_URL", "https://lebot.djiang.xyz")
SOURCE = os.environ.get("AUDIO_SOURCE") or _default_monitor()
RATE = 16000
PORT = 7788
CATS = ["BIOLOGY", "CHEMISTRY", "PHYSICS", "EARTH_SPACE", "MATH", "ENERGY", "OTHER"]
DG_URL = ("wss://api.deepgram.com/v1/listen?model=nova-2&encoding=linear16"
          f"&sample_rate={RATE}&channels=1&interim_results=true&punctuate=true"
          "&smart_format=true&endpointing=false")

_lock = threading.Lock()
_buf = bytearray()
state = {"running": False, "category": "OTHER", "gen": 0, "transcript": "", "thinking": "",
         "answering": False, "answer": None, "reasoning": "", "mode": "", "err": "",
         "refining": False, "steps": []}


def _capture():
    """Drain parec continuously; only buffer audio while a question is running."""
    p = subprocess.Popen(
        ["parec", "--format=s16le", f"--rate={RATE}", "--channels=1",
         "-d", SOURCE, "--latency-msec=50"], stdout=subprocess.PIPE)
    while True:
        chunk = p.stdout.read(1600)      # ~50ms
        if not chunk:
            break
        if state["running"]:
            with _lock:
                _buf.extend(chunk)


async def _think(client, text, my_gen):
    """During the read: fast tentative guess so you can watch it work the question out.
    No buzzing, no probabilities — just the current leaning."""
    try:
        d = (await client.post(f"{LEBOT_URL}/analyze", json={
            "prefix": text, "category": state["category"], "fast": True,
            "total_words": max(40, len(text.split()) * 2), "history": []})).json()
    except Exception:
        d = {}
    g = d.get("guess", "")
    with _lock:
        if state["gen"] == my_gen and g and g.upper() != "UNKNOWN":
            state["thinking"] = g


async def _full_call(client, text, cat):
    d = (await client.post(f"{LEBOT_URL}/analyze", json={
        "prefix": text, "category": cat, "fast": False,
        "total_words": len(text.split()), "history": []})).json()
    return d


PRECOMP_MIN = 12      # don't full-solve until the question is clearly under way
PRECOMP_STEP = 6      # words of new speech between rolling full-accuracy solves
END_TIMEOUT = 8.0     # hard cap on the final solve; fall back to best-so-far


async def _session(ws, client):
    """One question: stream mic PCM up, keep the clean transcript, and roll a full-accuracy
    solve on the growing text so the answer is ready the instant you press End."""
    my_gen = state["gen"]
    # cache of the latest full-accuracy solve, so End is usually instant
    pre = {"words": 0, "answer": None, "reasoning": "", "mode": ""}
    full_inflight = set()

    async def precompute(text, words):
        try:
            d = await _full_call(client, text, state["category"])
        except Exception:
            d = None
        if os.environ.get("LIVE_DEBUG"):
            print(f"  [precomp {words}w → {d.get('guess') if d else 'ERR'}]", flush=True)
        if d and state["gen"] == my_gen and words > pre["words"]:
            g, why = d.get("guess", "?"), d.get("reasoning", "")
            pre.update(words=words, answer=g, reasoning=why, mode=d.get("mode", ""))
            with _lock:  # record a thinking step whenever the working answer changes
                steps = state["steps"]
                if g and g.upper() != "UNKNOWN" and (not steps or steps[-1]["guess"] != g):
                    steps.append({"w": words, "guess": g, "why": why})
        full_inflight.discard(1)

    async def send():
        while state["running"] and state["gen"] == my_gen:
            with _lock:
                chunk = bytes(_buf)
                _buf.clear()
            if chunk:
                await ws.send(chunk)
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "CloseStream"}))

    sender = asyncio.create_task(send())
    finals, last_think, last_full, think_inflight = [], 0, 0, set()
    async for raw in ws:                  # runs until Deepgram closes the ws (after End)
        data = json.loads(raw)
        if data.get("type") != "Results":
            continue
        tr = data["channel"]["alternatives"][0].get("transcript", "")
        interim = "" if data.get("is_final") else tr
        if data.get("is_final") and tr:
            finals.append(tr)
        full = " ".join(finals + ([interim] if interim else [])).strip()
        with _lock:
            state["transcript"] = full
        nwords = len(full.split())
        if state["running"] and nwords > last_think and not think_inflight:   # live lean
            last_think = nwords
            think_inflight.add(1)
            t = asyncio.create_task(_think(client, full, my_gen))
            t.add_done_callback(lambda _: think_inflight.discard(1))
        if (state["running"] and nwords >= PRECOMP_MIN                        # roll a full solve
                and nwords >= last_full + PRECOMP_STEP and not full_inflight):
            last_full = nwords
            full_inflight.add(1)
            asyncio.create_task(precompute(full, nwords))
        if state["gen"] != my_gen:         # a new question started without a clean End
            break
    await sender
    if state["gen"] != my_gen:
        return
    # End: use the rolled solve if it already covers ~all of the question; else solve now (capped)
    final = state["transcript"]
    fw = len(final.split())
    # Instant path: a confident rolled solve covering ~most of the question (the last few
    # words rarely change a recall answer). Show it now, then silently re-solve the complete
    # text in the background and correct if it actually differs — speed without losing accuracy.
    have_pre = (pre["answer"] and pre["answer"].upper() != "UNKNOWN"
                and pre["words"] >= max(PRECOMP_MIN, int(0.6 * fw)))
    if os.environ.get("LIVE_DEBUG"):
        print(f"  [END fw={fw} pre={pre['words']}w/{pre['answer']} have_pre={have_pre}]", flush=True)
    if have_pre:
        # Show the rolled answer instantly, then re-solve the complete text and firm it up.
        with _lock:
            if state["gen"] == my_gen:
                state.update(answer=pre["answer"], reasoning=pre["reasoning"],
                             mode=pre["mode"], answering=False, refining=True)

        async def verify():
            try:
                d = await _full_call(client, final, state["category"])
            except Exception:
                d = None
            g = d.get("guess", "") if d else ""
            with _lock:
                if state["gen"] == my_gen:
                    if g and g.upper() != "UNKNOWN":
                        state.update(answer=g, reasoning=d.get("reasoning", ""), mode=d.get("mode", ""))
                    state["refining"] = False
        asyncio.create_task(verify())
        return

    try:
        d = await asyncio.wait_for(_full_call(client, final, state["category"]), END_TIMEOUT)
    except Exception:
        d = ({"guess": pre["answer"], "reasoning": pre["reasoning"] + " (from just before the end)",
              "mode": pre["mode"]} if pre["answer"]
             else {"guess": state.get("thinking") or "?", "reasoning": "(timed out — best guess)",
                   "mode": ""})
    with _lock:
        if state["gen"] == my_gen:
            state.update(answer=d.get("guess", "?"), reasoning=d.get("reasoning", ""),
                         mode=d.get("mode", ""), answering=False)


async def _dg_loop():
    if not DG_KEY:
        with _lock:
            state["err"] = "DEEPGRAM_API_KEY missing — add it to .env"
        return
    client = httpx.AsyncClient(timeout=40)
    while True:
        if not state["running"]:
            await asyncio.sleep(0.1)
            continue
        try:
            async with websockets.connect(
                    DG_URL, additional_headers={"Authorization": f"Token {DG_KEY}"}) as ws:
                await _session(ws, client)
        except Exception as e:
            with _lock:
                state["err"] = f"deepgram: {str(e)[:80]}"
            await asyncio.sleep(0.5)


def _start(category):
    with _lock:
        _buf.clear()
        state.update(category=category, transcript="", thinking="", answer=None, reasoning="",
                     mode="", err="", answering=False, refining=False, steps=[],
                     gen=state["gen"] + 1, running=True)


def _stop():
    with _lock:
        state["running"] = False
        state["answering"] = True         # the Deepgram session flushes + calls _answer


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LeBot live</title><style>
body{background:#1a1b26;color:#c0caf5;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:760px}
h1{color:#7aa2f7;font-size:20px;margin:0 0 4px}.sub{color:#565f89;margin-bottom:16px}
button{font:600 16px inherit;border:0;border-radius:10px;padding:14px 22px;cursor:pointer;margin-right:8px}
#start{background:#9ece6a;color:#1a1b26}#stop{background:#f7768e;color:#1a1b26}
button:disabled{opacity:.35;cursor:default}
select{background:#24283b;color:#c0caf5;border:1px solid #2f334d;border-radius:8px;padding:10px;font:inherit;margin-left:8px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.live{background:#9ece6a;box-shadow:0 0 8px #9ece6a}.idle{background:#565f89}
.now{background:#24283b;border:1px solid #2f334d;border-radius:10px;padding:16px;margin:16px 0}
.q{color:#9aa3b2;font-size:14px;min-height:20px}
.think{color:#7aa2f7;font-size:15px;margin-top:8px;min-height:20px}.think b{color:#bb9af7}
.steps{margin-top:14px}
.step{border-left:2px solid #2f334d;padding:6px 0 6px 12px;margin:0 0 6px}
.step:last-child{border-left-color:#bb9af7}
.stepw{color:#565f89;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.stepg{color:#bb9af7;font-weight:600}.stepy{color:#9aa3b2;font-size:13px;font-style:italic;margin-top:2px}
.slabel{color:#565f89;font-size:12px;text-transform:uppercase;letter-spacing:.05em;margin-top:6px}
.card{background:#2a2e45;border:1px solid #bb9af7;border-radius:10px;padding:18px;margin:16px 0}
.alabel{color:#565f89;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.answer{font-size:30px;color:#9ece6a;font-weight:700;margin:4px 0 10px}
.why{color:#c0caf5;line-height:1.6}.mode{color:#565f89;font-size:12px;margin-top:8px}
.err{color:#f7768e;font-size:13px;margin-top:8px}
</style>
<h1>LeBot — live answerer</h1>
<div class=sub>Press Start when the reader begins, End when the question finishes. Answer appears on End.</div>
<div>
  <button id=start onclick=start()>▶ Question Start</button>
  <button id=stop onclick=stop() disabled>■ Question End</button>
  <select id=cat>__CATS__</select>
</div>
<div class=now>
  <div><span id=dot class="dot idle"></span><span id=status>idle</span></div>
  <div class=q id=q style=margin-top:8px></div>
  <div class=think id=think></div>
  <div class=slabel id=slabel style=display:none>How it's thinking</div>
  <div class=steps id=steps></div>
</div>
<div class=card id=card style=display:none>
  <div class=alabel>Answer</div>
  <div class=answer id=answer>—</div>
  <div class=why id=why></div>
  <div class=mode id=mode></div>
</div>
<div class=err id=err></div>
<script>
function $(s){return document.getElementById(s)}
async function start(){await fetch('/start?cat='+$('cat').value,{method:'POST'})}
async function stop(){await fetch('/stop',{method:'POST'})}
async function tick(){
  let s;try{s=await (await fetch('/state')).json()}catch(e){return}
  $('start').disabled=s.running||s.answering;$('stop').disabled=!s.running
  $('dot').className='dot '+(s.running?'live':'idle')
  $('status').textContent=s.running?('listening · '+s.category):(s.answering?'thinking…':'idle')
  $('q').textContent=s.transcript||''
  $('think').innerHTML=(s.running&&s.thinking)?('leaning toward <b>'+s.thinking+'</b>…'):''
  let steps=s.steps||[]
  $('slabel').style.display=steps.length?'block':'none'
  $('steps').innerHTML=steps.map(st=>'<div class=step><span class=stepw>heard '+st.w+' words</span>'
    +'<div class=stepg>'+st.guess+'</div>'+(st.why?'<div class=stepy>'+st.why+'</div>':'')+'</div>').join('')
  $('err').textContent=s.err||''
  let c=$('card')
  if(s.answering){c.style.display='block';$('answer').textContent='…';$('why').textContent='';$('mode').textContent=''}
  else if(s.answer){c.style.display='block';$('answer').textContent=s.answer;$('why').textContent=s.reasoning||'';$('mode').textContent=(s.refining?'refining…':(s.mode?('mode: '+s.mode):''))}
  else if(s.running){c.style.display='none'}
}
setInterval(tick,400);tick()
</script>"""
PAGE = PAGE.replace("__CATS__", "".join(f"<option>{c}</option>" for c in CATS))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE)
        elif self.path == "/state":
            with _lock:
                self._send(200, json.dumps(state), "application/json")
        else:
            self._send(404, "no")

    def do_POST(self):
        if self.path.startswith("/start"):
            cat = self.path.split("cat=", 1)[1].split("&")[0].upper() if "cat=" in self.path else "OTHER"
            _start(cat if cat in CATS else "OTHER")
            self._send(200, "{}", "application/json")
        elif self.path == "/stop":
            _stop()
            self._send(200, "{}", "application/json")
        else:
            self._send(404, "no")


def main():
    print(f"capture source: {SOURCE}", flush=True)
    print("Deepgram key: " + ("set" if DG_KEY else "MISSING (add DEEPGRAM_API_KEY to .env)"), flush=True)
    threading.Thread(target=_capture, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(_dg_loop()), daemon=True).start()
    print(f"open http://localhost:{PORT}", flush=True)
    if not os.environ.get("NO_OPEN"):
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
        except Exception:
            pass
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
