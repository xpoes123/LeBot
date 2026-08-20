"""LeBot live tester — a tiny local web app for manually-bounded question testing.

You bound each question with ▶ Start / ■ End. Between them the app captures your computer
audio (default sink monitor) and streams it to Deepgram for clean real-time transcription;
each new word is fed to /analyze so you see the answer trajectory live, and on End it freezes
the earliness summary ('settled at word N/M'). Manual bounds = honest per-question timing,
and Deepgram (context-aware) avoids the garbling that isolated-chunk Whisper produced.

Run:  ./botvenv/bin/python live.py     then open http://localhost:7788
Env:  DEEPGRAM_API_KEY (required)   AUDIO_SOURCE (default: default sink monitor)
      LEBOT_URL (default VPS)
Nothing is stored; audio only leaves your machine to Deepgram while a question is running.
"""
import asyncio
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import numpy as np
import websockets

from listen import _default_monitor, _settle_word

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
state = {"running": False, "category": "OTHER", "text": "", "p": 0.0, "gen": 0,
         "traj": [], "history": [], "final": None, "settle": 0, "total": 0,
         "err": "", "transcript": "", "scoring": False}


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


async def _analyze(client, text, nwords):
    """Live during-question call — just to show a rough answer as the reader speaks."""
    try:
        d = (await client.post(f"{LEBOT_URL}/analyze", json={
            "prefix": text, "category": state["category"], "fast": True,
            "total_words": max(40, nwords * 2), "history": state["history"]})).json()
    except Exception:
        d = {}
    guess = d.get("guess", "?")
    with _lock:
        state["text"] = text
        state["p"] = d.get("p_buzz", 0.0)
        state["history"].append({"guess": guess, "mode": d.get("mode", "recall")})


async def _endpass(client, full, my_gen):
    """On End: replay ~10 evenly-spaced prefixes of the COMPLETE clean transcript through
    /analyze in parallel, then compute the honest 'settled at word N/M'. This is the number
    to trust — the live samples are too sparse to be reliable."""
    words = full.split()
    n = len(words)
    if not n:
        with _lock:
            state["scoring"] = False
        return
    K = 10
    ks = sorted({max(1, round(n * i / K)) for i in range(1, K + 1)})

    async def one(k):
        try:
            d = (await client.post(f"{LEBOT_URL}/analyze", json={
                "prefix": " ".join(words[:k]), "category": state["category"], "fast": True,
                "total_words": n, "history": []})).json()
        except Exception:
            d = {}
        return [k, d.get("guess", "?")]

    traj = await asyncio.gather(*[one(k) for k in ks])
    final, settle, total = _settle_word(traj)
    with _lock:
        if state["gen"] != my_gen:          # a new question already started — discard
            return
        state.update(traj=traj, final=final, settle=settle, total=total, scoring=False)


async def _session(ws, client):
    """One question: stream mic PCM up, assemble the transcript, analyze on each new word."""
    my_gen = state["gen"]

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
    finals, last_words, inflight = [], 0, set()
    async for raw in ws:                     # runs until Deepgram closes the ws (after End)
        data = json.loads(raw)
        if data.get("type") != "Results":
            continue
        alt = data["channel"]["alternatives"][0]
        tr = alt.get("transcript", "")
        interim = "" if data.get("is_final") else tr
        if data.get("is_final") and tr:
            finals.append(tr)
        full = " ".join(finals + ([interim] if interim else [])).strip()
        with _lock:
            state["transcript"] = full
        nwords = len(full.split())
        if state["running"] and nwords > last_words and not inflight:
            last_words = nwords
            inflight.add(1)
            t = asyncio.create_task(_analyze(client, full, nwords))
            t.add_done_callback(lambda _: inflight.discard(1))
        if state["gen"] != my_gen:            # a new question started without a clean End
            break
    await sender
    if state["gen"] == my_gen:                # question ended — score the full clean transcript
        await _endpass(client, state["transcript"], my_gen)


async def _dg_loop():
    """Wait for a question, open a fresh Deepgram stream for it, tear down on End."""
    if not DG_KEY:
        with _lock:
            state["err"] = "DEEPGRAM_API_KEY missing — add it to .env"
        return
    client = httpx.AsyncClient(timeout=20)
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
        state.update(category=category, text="", p=0.0, traj=[], history=[], transcript="",
                     final=None, settle=0, total=0, err="", scoring=False,
                     gen=state["gen"] + 1, running=True)


def _stop():
    # The Deepgram session flushes the final transcript and runs _endpass to compute the
    # trustworthy metric; here we just stop capture and mark that scoring is under way.
    with _lock:
        state["running"] = False
        state["scoring"] = True


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LeBot live</title><style>
body{background:#1a1b26;color:#c0caf5;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:760px}
h1{color:#7aa2f7;font-size:20px;margin:0 0 4px}.sub{color:#565f89;margin-bottom:16px}
button{font:600 16px inherit;border:0;border-radius:10px;padding:14px 22px;cursor:pointer;margin-right:8px}
#start{background:#9ece6a;color:#1a1b26}#stop{background:#f7768e;color:#1a1b26}
button:disabled{opacity:.35;cursor:default}
select{background:#24283b;color:#c0caf5;border:1px solid #2f334d;border-radius:8px;padding:10px;font:inherit;margin-left:8px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.live{background:#9ece6a;box-shadow:0 0 8px #9ece6a}.idle{background:#565f89}
.now{background:#24283b;border:1px solid #2f334d;border-radius:10px;padding:16px;margin:16px 0}
.guess{font-size:26px;color:#bb9af7;font-weight:700}.p{color:#e0af68;font-variant-numeric:tabular-nums}
.heard{color:#9aa3b2;font-size:13px;margin-top:8px;min-height:18px}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}
td{padding:3px 8px;border-bottom:1px solid #2f334d}.w{color:#565f89;width:60px}
.final{background:#2a2e45;border:1px solid #bb9af7;border-radius:10px;padding:16px;margin:16px 0;font-size:18px}
.final b{color:#9ece6a}.err{color:#f7768e;font-size:13px;margin-top:8px}
</style>
<h1>LeBot — live question tester</h1>
<div class=sub>Press Start when the reader begins, End when they finish. Audio streams to Deepgram only while running.</div>
<div>
  <button id=start onclick=start()>▶ Question Start</button>
  <button id=stop onclick=stop() disabled>■ Question End</button>
  <select id=cat>__CATS__</select>
</div>
<div class=now>
  <div><span id=dot class="dot idle"></span><span id=status>idle</span></div>
  <div style=margin-top:8px><span class=guess id=guess>—</span> &nbsp;<span class=p id=p></span></div>
  <div class=heard id=heard></div>
  <table id=traj></table>
  <div class=err id=err></div>
</div>
<div class=final id=final style=display:none></div>
<script>
function $(s){return document.getElementById(s)}
async function start(){await fetch('/start?cat='+$('cat').value,{method:'POST'})}
async function stop(){await fetch('/stop',{method:'POST'})}
async function tick(){
  let s;try{s=await (await fetch('/state')).json()}catch(e){return}
  $('start').disabled=s.running||s.scoring;$('stop').disabled=!s.running
  $('dot').className='dot '+(s.running?'live':'idle')
  $('status').textContent=s.running?('listening · '+s.category):(s.scoring?'scoring…':'idle')
  $('guess').textContent=s.traj.length?s.traj[s.traj.length-1][1]:'—'
  $('p').textContent=(s.running&&s.traj.length)?('P='+(s.p*100|0)+'%'):''
  $('heard').textContent=s.transcript?('… '+s.transcript.slice(-160)):''
  $('traj').innerHTML=s.traj.slice(-12).map(r=>'<tr><td class=w>'+r[0]+'w</td><td>'+r[1]+'</td></tr>').join('')
  $('err').textContent=s.err||''
  let f=$('final')
  if(s.scoring){f.style.display='block';f.innerHTML='scoring the full transcript…'}
  else if(!s.running && s.final){f.style.display='block';f.innerHTML='── FINAL: <b>'+s.final+'</b> &nbsp; settled at word '+s.settle+'/'+s.total+' ('+(s.total?100*s.settle/s.total|0:0)+'% in)'}
  else if(s.running){f.style.display='none'}
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
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
