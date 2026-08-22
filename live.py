"""LeBot live answerer — a tiny local web app. No bot, no buzzing, no manual clicking.

It listens continuously to your computer audio (streamed to Deepgram), recognises when a
Science Bowl question STARTS from the moderator's announcement ("toss-up, biology, short
answer" -> category + format for free), works the answer out as the question is read, and
finishes when the reader pauses (a completed read OR a buzz interrupt). Manual ▶/■ buttons
remain as an override. Deepgram is fed Science-Bowl vocabulary (W/X/Y/Z, toss-up, bonus,
categories) so it stops mishearing the MC letters and the announcements.

Run:  ./botvenv/bin/python live.py     (auto-opens http://localhost:7788)
Env:  DEEPGRAM_API_KEY (required)   AUDIO_SOURCE (default: default sink monitor)
      LEBOT_URL (default VPS)   NO_OPEN=1 to skip opening the browser
"""
import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
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

# Science Bowl format knowledge fed to Deepgram so it stops mishearing the spoken MC
# letters (Z<->D) and the announcements. keyword:intensifier boosts recognition.
_KEYWORDS = ["W:5", "X:5", "Y:5", "Z:5", "toss-up:3", "tossup:3", "bonus:3",
             "short answer:3", "multiple choice:3", "biology:2", "chemistry:2",
             "physics:2", "math:2", "mathematics:2", "energy:2", "earth and space:2",
             "interrupt:4", "correct:2", "incorrect:2"]
_PARAMS = [("model", "nova-2"), ("encoding", "linear16"), ("sample_rate", str(RATE)),
           ("channels", "1"), ("interim_results", "true"), ("punctuate", "true"),
           ("smart_format", "true"), ("endpointing", "800")]  # ~0.8s pause = end of read (snappier)
DG_URL = ("wss://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(_PARAMS)
          + "".join("&keywords=" + urllib.parse.quote(k) for k in _KEYWORDS))

_CATMAP = {"physic": "PHYSICS", "math": "MATH", "chem": "CHEMISTRY", "bio": "BIOLOGY",
           "energy": "ENERGY", "earth": "EARTH_SPACE", "space": "EARTH_SPACE"}
# Anchor on "<category> short answer|multiple choice" — the moderator always says it, and
# Deepgram often mangles "toss-up"/"bonus" (heard "Possibilities of"). Everything AFTER the
# format phrase is the question; the category word is right before it. group(1)=cat group(2)=fmt.
_CATWORD = r"(biolog\w*|chem\w*|physic\w*|math\w*|energy|earth(?:\s+and)?(?:\s+space)?|space)"
_MARK = re.compile(_CATWORD + r"[\s,.]+(?:and\s+)?(short answer|multiple choice)", re.I)
# The read is over the moment the moderator/players react — SB audio rarely leaves a pause,
# so these content cues, not silence, are the real end-of-question signal.
# end-of-read cues. "interrupt" only counts as a real buzz when it looks like the moderator
# confirming one ("Peter, interrupt?" / "Vish. Interrupt.") — not mid-sentence STT garble
# ("...the following interrupt that..."), which was ending questions early.
_CONF = re.compile(r"that(?:'s| is) (?:in)?correct|i'?ll reread|"
                   r"\w+[,.]\s+interrupt\b|\binterrupt\s*\?", re.I)
# a multiple-choice read is done once all four option letters have gone by, in order
_OPTS = re.compile(r"\bw\b.{0,160}\bx\b.{0,160}\by\b.{0,160}\bz\b", re.I | re.S)
MAX_Q_WORDS = 90      # a runaway guard: no real question runs this long


def _catof(s):
    s = s.lower()
    for k, v in _CATMAP.items():
        if k in s:
            return v
    return "OTHER"


_lock = threading.Lock()
_buf = bytearray()
state = {"mode": "waiting",      # waiting | reading | answering
         "category": "", "qformat": "", "transcript": "", "thinking": "",
         "answer": None, "reasoning": "", "resmode": "", "steps": [], "refining": False,
         "err": "", "gen": 0, "force_start": False, "force_end": False, "force_clear": False,
         "cat_override": "", "log": []}


def _log_locked(q, ans, why, mode, cat):
    """Append a finished question to the round log (caller holds _lock). Skip abstentions."""
    if not ans or ans.upper() == "UNKNOWN":
        return
    state["log"].insert(0, {"q": q, "answer": ans, "why": why, "mode": mode, "cat": cat})
    del state["log"][40:]

PRECOMP_MIN = 8       # start trying an answer once the question is a bit under way
PRECOMP_STEP = 5      # words of new speech between full-accuracy answer attempts
THINK_STEP = 3        # words between fast "leaning" updates (keep a fresh answer ready for End)
END_TIMEOUT = 18.0    # hard cap on the final solve (Opus-retry math can be ~12s); then fall back


def _capture():
    """Continuously buffer computer audio — we listen the whole time, not per-question."""
    p = subprocess.Popen(
        ["parec", "--format=s16le", f"--rate={RATE}", "--channels=1",
         "-d", SOURCE, "--latency-msec=50"], stdout=subprocess.PIPE)
    while True:
        chunk = p.stdout.read(1600)
        if not chunk:
            break
        with _lock:
            _buf.extend(chunk)


async def _think(client, text, gen):
    try:
        d = (await client.post(f"{LEBOT_URL}/analyze", json={
            "prefix": text, "category": state["category"] or "OTHER", "fast": True,
            "total_words": max(40, len(text.split()) * 2), "history": []})).json()
    except Exception:
        d = {}
    g = d.get("guess", "")
    with _lock:
        if state["gen"] == gen and g and g.upper() != "UNKNOWN":
            state["thinking"] = g


async def _full_call(client, text, cat):
    d = (await client.post(f"{LEBOT_URL}/analyze", json={
        "prefix": text, "category": cat or "OTHER", "fast": False,
        "total_words": len(text.split()), "history": []})).json()
    return d


def _begin(cat, form, text0):
    with _lock:
        state.update(mode="reading", category=(state["cat_override"] or cat), qformat=form,
                     transcript=text0, thinking="", answer=None, reasoning="", resmode="",
                     steps=[], refining=False, gen=state["gen"] + 1)
        return state["gen"]


async def _run(ws, client):
    """One long-lived Deepgram connection: detect starts, read, answer on pause, repeat."""
    async def send():
        while True:
            with _lock:
                chunk = bytes(_buf)
                _buf.clear()
            if chunk:
                await ws.send(chunk)
            await asyncio.sleep(0.05)

    sender = asyncio.create_task(send())
    finals, interim = [], ""
    reading, my_gen, qcat = False, 0, "OTHER"
    last_think, last_full = 0, 0
    think_inflight, full_inflight = set(), set()
    pre = {"words": 0, "answer": None, "reasoning": "", "mode": ""}

    def reset_reading():
        nonlocal reading, finals, interim, last_think, last_full, pre
        reading, finals, interim, last_think, last_full = False, [], "", 0, 0
        pre = {"words": 0, "answer": None, "reasoning": "", "mode": ""}

    async def precompute(text, words, gen, cat):
        try:
            d = await _full_call(client, text, cat)
        except Exception:
            d = None
        if os.environ.get("LIVE_DEBUG"):
            print(f"  [precomp {words}w → {d.get('guess') if d else 'ERR'}]", flush=True)
        if d and state["gen"] == gen and words > pre["words"]:
            g, why = d.get("guess", "?"), d.get("reasoning", "")
            pre.update(words=words, answer=g, reasoning=why, mode=d.get("mode", ""))
            with _lock:
                steps = state["steps"]
                if g and g.upper() != "UNKNOWN" and (not steps or steps[-1]["guess"] != g):
                    steps.append({"w": words, "guess": g, "why": why})
        full_inflight.discard(1)

    async def answer_now(qtext, gen, cat, interrupt=False):
        # Show a quick answer INSTANTLY the moment the question ends — the confident rolled
        # solve if we have one, else the live lean — then finalize with a considered solve in
        # the background (which may correct it). This is what makes the answer appear on End
        # instead of after a multi-second solve.
        quick = pre["answer"] if (pre["answer"] and pre["answer"].upper() != "UNKNOWN") \
            else state.get("thinking")
        have_pre_reason = bool(quick and quick == pre["answer"])
        with _lock:
            if state["gen"] == gen:
                state.update(answer=(quick or "…"),
                             reasoning=(pre["reasoning"] if have_pre_reason else ""),
                             resmode=(pre["mode"] if have_pre_reason else ""),
                             refining=True, mode="waiting")

        async def finalize():
            try:
                d = await asyncio.wait_for(_full_call(client, qtext, cat), END_TIMEOUT)
            except Exception:
                d = None
            g = (d or {}).get("guess", "")
            with _lock:
                if state["gen"] != gen:
                    return
                if g and g.upper() != "UNKNOWN":
                    state.update(answer=g, reasoning=d.get("reasoning", ""), resmode=d.get("mode", ""))
                elif not quick:  # nothing shown and nothing found
                    state.update(answer="UNKNOWN",
                                 reasoning=(d.get("reasoning", "") if d else "(no answer)"), resmode="")
                # else: keep the quick answer we already showed
                state["refining"] = False
                _log_locked(qtext, state["answer"], state["reasoning"], state["resmode"], cat)
        asyncio.create_task(finalize())

    async for raw in ws:
        data = json.loads(raw)
        if data.get("type") != "Results":
            continue
        alt = data["channel"]["alternatives"][0]
        tr = alt.get("transcript", "")
        is_final = data.get("is_final")
        speech_final = data.get("speech_final")
        if is_final:
            if tr:
                finals.append(tr)
            interim = ""
        else:
            interim = tr

        force_start = state["force_start"]
        force_end = state["force_end"]
        force_clear = state["force_clear"]
        if force_start or force_end or force_clear:
            with _lock:
                state["force_start"] = state["force_end"] = state["force_clear"] = False

        if force_clear:                       # manual: drop the current read, back to listening
            reset_reading()
            with _lock:
                state.update(mode="waiting", transcript="", thinking="", steps=[])
            continue

        if not reading:
            joined = " ".join(finals).strip()
            m = _MARK.search(joined)
            if m or force_start:
                qcat = _catof(m.group(1)) if m else "OTHER"
                form = m.group(2).lower() if m else ""
                after = joined[m.end():].strip() if m else ""
                my_gen = _begin(qcat, form, after)
                reading, finals, interim = True, ([after] if after else []), ""
                last_think, last_full = 0, 0
                pre = {"words": 0, "answer": None, "reasoning": "", "mode": ""}
        else:
            joined = " ".join(finals)
            qtext = (joined + " " + interim).strip()
            with _lock:
                state["transcript"] = qtext
            nwords = len(qtext.split())
            if nwords >= last_think + THINK_STEP and not think_inflight:
                last_think = nwords
                think_inflight.add(1)
                t = asyncio.create_task(_think(client, qtext, my_gen))
                t.add_done_callback(lambda _: think_inflight.discard(1))
            if (nwords >= PRECOMP_MIN and nwords >= last_full + PRECOMP_STEP
                    and not full_inflight):
                last_full = nwords
                full_inflight.add(1)
                asyncio.create_task(precompute(qtext, nwords, my_gen, state["category"]))

            # End of read, by question grammar (silence is unreliable in SB audio):
            #  · short answer ends at the "?"
            #  · multiple choice ends once all four options W/X/Y/Z have been read
            #  · the next announcement, or a correct/incorrect cue, ends it too (and the
            #    next announcement also starts that question); a word cap backstops runaways.
            is_mc = "multiple" in state["qformat"]
            m2 = _MARK.search(joined)
            conf = _CONF.search(qtext)
            end_q, restart, interrupt = None, None, False
            if m2:
                end_q = joined[:m2.start()].strip()
                restart = (_catof(m2.group(1)), m2.group(2).lower(), joined[m2.end():].strip())
            elif conf:
                end_q = qtext[:conf.start()].strip()
                interrupt = "interrupt" in conf.group(0).lower()  # other team buzzed — commit now
            elif not is_mc and nwords >= 6 and "?" in joined:
                end_q = joined[:joined.index("?") + 1].strip()
            elif (not is_mc and speech_final and nwords >= 6
                  and joined.rstrip().endswith((".", "?", "!"))):
                # reader paused after a COMPLETE sentence (not a mid-clause comma) — end here.
                # The sentence-punctuation guard stops a mid-question breath cutting it off.
                end_q = qtext
            elif is_mc and _OPTS.search(joined):
                end_q = qtext
            elif (is_mc and speech_final and nwords >= 12
                  and joined.rstrip().endswith((".", "?", "!"))):
                # options can be garbled (math symbols, mis-heard Z) so _OPTS won't match; a
                # pause after a completed sentence means the reader finished — end before chatter
                end_q = qtext
            elif force_end or nwords >= MAX_Q_WORDS:
                end_q = qtext

            if end_q is not None:
                if len(end_q.split()) >= 3:
                    await answer_now(end_q, my_gen, state["category"], interrupt)
                else:
                    with _lock:
                        state["mode"] = "waiting"
                reset_reading()
                if restart:
                    cat2, form2, after2 = restart
                    my_gen = _begin(cat2, form2, after2)
                    reading, finals, interim = True, ([after2] if after2 else []), ""
                    last_think, last_full = 0, 0
                    pre = {"words": 0, "answer": None, "reasoning": "", "mode": ""}

    await sender


async def _stream():
    if not DG_KEY:
        with _lock:
            state["err"] = "DEEPGRAM_API_KEY missing — add it to .env"
        return
    client = httpx.AsyncClient(timeout=40)
    while True:
        try:
            async with websockets.connect(
                    DG_URL, additional_headers={"Authorization": f"Token {DG_KEY}"}) as ws:
                await _run(ws, client)
        except Exception as e:
            with _lock:
                state["err"] = f"deepgram: {str(e)[:80]}"
            await asyncio.sleep(0.5)


PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>LeBot live</title><style>
body{background:#1a1b26;color:#c0caf5;font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;max-width:760px}
h1{color:#7aa2f7;font-size:20px;margin:0 0 4px}.sub{color:#565f89;margin-bottom:16px}
button{font:600 14px inherit;border:0;border-radius:9px;padding:10px 16px;cursor:pointer;margin-right:8px}
#start{background:#414868;color:#c0caf5}#stop{background:#414868;color:#c0caf5}
button:disabled{opacity:.35;cursor:default}
select{background:#24283b;color:#c0caf5;border:1px solid #2f334d;border-radius:8px;padding:8px;font:inherit;margin-left:8px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.reading{background:#9ece6a;box-shadow:0 0 8px #9ece6a}.wait{background:#565f89}.ans{background:#e0af68}
.badge{font-size:12px;padding:2px 8px;border-radius:6px;background:#414868;color:#c0caf5;margin-left:8px}
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
.logh{color:#7aa2f7;font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin:24px 0 6px;border-top:1px solid #2f334d;padding-top:14px}
.le{background:#1f2335;border:1px solid #2f334d;border-radius:9px;padding:12px 14px;margin:8px 0}
.leh{font-size:12px;color:#565f89}.lea{color:#9ece6a;font-weight:700;font-size:17px;margin:2px 0}
.leq{color:#9aa3b2;font-size:13px;margin:4px 0}.lew{color:#a9b1d6;font-size:13px;line-height:1.5}
</style>
<h1>LeBot — live answerer</h1>
<div class=sub>Listening. It starts itself when it hears "toss-up … short answer", answers when the reader pauses. Buttons are a manual override.</div>
<div>
  <button id=start onclick=fstart()>▶ Force start</button>
  <button id=stop onclick=fstop()>■ Answer now</button>
  <button id=clear onclick=fclear()>✕ Clear</button>
  <label style="color:#565f89;font-size:13px;margin-left:8px">category
  <select id=cat onchange=setcat()>
    <option value="">auto</option>__CATS__
  </select></label>
</div>
<div class=now>
  <div><span id=dot class="dot wait"></span><span id=status>listening…</span><span class=badge id=cb style=display:none></span></div>
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
<div class=logh id=logh style=display:none>Answered this round</div>
<div id=log></div>
<script>
function $(s){return document.getElementById(s)}
function esc(t){return String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function fstart(){await fetch('/start',{method:'POST'})}
async function fstop(){await fetch('/stop',{method:'POST'})}
async function fclear(){await fetch('/clear',{method:'POST'})}
async function setcat(){await fetch('/cat?c='+$('cat').value,{method:'POST'})}
async function tick(){
  let s;try{s=await (await fetch('/state')).json()}catch(e){return}
  let reading=s.mode==='reading', answering=s.mode==='answering'
  $('dot').className='dot '+(reading?'reading':(answering?'ans':'wait'))
  $('status').textContent=reading?'reading a question…':(answering?'answering…':'listening…')
  $('cb').style.display=(reading&&s.category)?'inline':'none'
  $('cb').textContent=s.category+(s.qformat?(' · '+s.qformat):'')
  $('q').textContent=s.transcript||''
  $('think').innerHTML=(reading&&s.thinking)?('leaning toward <b>'+esc(s.thinking)+'</b>…'):''
  let steps=s.steps||[]
  $('slabel').style.display=steps.length?'block':'none'
  $('steps').innerHTML=steps.map(st=>'<div class=step><span class=stepw>heard '+(st.w|0)+' words</span>'
    +'<div class=stepg>'+esc(st.guess)+'</div>'+(st.why?'<div class=stepy>'+esc(st.why)+'</div>':'')+'</div>').join('')
  $('err').textContent=s.err||''
  let c=$('card')
  if(answering){c.style.display='block';$('answer').textContent='…';$('why').textContent='';$('mode').textContent=''}
  else if(s.answer){c.style.display='block';$('answer').textContent=s.answer;$('why').textContent=s.reasoning||'';$('mode').textContent=(s.refining?'refining…':(s.resmode?('mode: '+s.resmode):''))}
  else{c.style.display='none'}
  let lg=s.log||[]
  $('logh').style.display=lg.length?'block':'none'
  $('log').innerHTML=lg.map(e=>'<div class=le><div class=leh>'+esc(e.cat)+(e.mode?(' · '+esc(e.mode)):'')+'</div>'
    +'<div class=lea>'+esc(e.answer)+'</div>'
    +'<div class=leq>'+esc(e.q)+'</div>'
    +(e.why?'<div class=lew>'+esc(e.why)+'</div>':'')+'</div>').join('')
}
setInterval(tick,400);tick()
</script>"""
PAGE = PAGE.replace("__CATS__", "".join(f'<option value="{c}">{c}</option>' for c in CATS))


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
            with _lock:
                state["force_start"] = True
            self._send(200, "{}", "application/json")
        elif self.path == "/stop":
            with _lock:
                state["force_end"] = True
            self._send(200, "{}", "application/json")
        elif self.path == "/clear":
            with _lock:
                state["force_clear"] = True
            self._send(200, "{}", "application/json")
        elif self.path.startswith("/cat"):
            c = self.path.split("c=", 1)[1].split("&")[0].upper() if "c=" in self.path else ""
            with _lock:
                state["cat_override"] = c if c in CATS else ""
            self._send(200, "{}", "application/json")
        else:
            self._send(404, "no")


def _free_port(port):
    """Kill any process (a stale live.py) still holding our port, so a relaunch always works.
    ss only reports pids for this user's own sockets, so we won't touch anything else."""
    try:
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        out = ""
    for pid in {int(p) for p in re.findall(r"pid=(\d+)", out)}:
        if pid != os.getpid():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def main():
    print(f"capture source: {SOURCE}", flush=True)
    print("Deepgram key: " + ("set" if DG_KEY else "MISSING (add DEEPGRAM_API_KEY to .env)"), flush=True)
    srv = None
    for attempt in range(6):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
            break
        except OSError:
            if attempt == 0:
                print(f"port {PORT} busy — clearing a stale instance…", flush=True)
            _free_port(PORT)
            time.sleep(0.5)
    if srv is None:
        print(f"could not bind {PORT}. Run:  fuser -k {PORT}/tcp", flush=True)
        return
    threading.Thread(target=_capture, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(_stream()), daemon=True).start()
    print(f"open http://localhost:{PORT}", flush=True)
    if not os.environ.get("NO_OPEN"):
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
        except Exception:
            pass
    srv.serve_forever()


if __name__ == "__main__":
    main()
