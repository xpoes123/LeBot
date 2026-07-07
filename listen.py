"""LeBot local listener — captures the audio you HEAR (the reader in a Discord call)
straight from the PipeWire monitor, transcribes on the GPU, and runs the proven
transcript->/analyze->buzz pipeline. No Discord voice API (which is broken against
Discord's current protocol) — just the sound coming out of your headset.

Usage:
  ./botvenv/bin/python listen.py [category]
Env:
  AUDIO_SOURCE   PipeWire source to capture (default: the default sink's monitor)
  LEBOT_URL      anticipation endpoint (default https://lebot.djiang.xyz)

A ~2.5s silence gap ends the current question and resets for the next one.
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from faster_whisper import WhisperModel

def _load_env():
    for line in Path(__file__).with_name(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
LEBOT_URL = os.environ.get("LEBOT_URL", "https://lebot.djiang.xyz")
CATEGORY = (sys.argv[1].upper() if len(sys.argv) > 1 else "OTHER")
RATE = 16000


def post_discord(msg):
    tok, ch = os.environ.get("DISCORD_TOKEN"), os.environ.get("DISCORD_CHANNEL_ID")
    if not tok or not ch:
        return
    try:
        httpx.post(f"https://discord.com/api/v10/channels/{ch}/messages",
                   headers={"Authorization": f"Bot {tok}"},
                   json={"content": msg[:1900]}, timeout=10)
    except Exception as e:
        print("discord post err:", e, flush=True)


def _default_monitor():
    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip()
    return f"{sink}.monitor"


SOURCE = os.environ.get("AUDIO_SOURCE") or _default_monitor()


def _load_whisper():
    import ctypes
    import glob
    try:
        import nvidia
        base = list(nvidia.__path__)[0]
        for so in sorted(glob.glob(os.path.join(base, "*", "lib", "*.so*"))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        m = WhisperModel("small.en", device="cuda", compute_type="float16")
        print("Whisper on GPU (RTX 4060)", flush=True)
        return m
    except Exception as e:
        print(f"GPU unavailable ({str(e)[:60]}); CPU", flush=True)
        return WhisperModel("small.en", device="cpu", compute_type="int8")


_model = _load_whisper()


def _transcribe(audio):
    segs, _ = _model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text for s in segs).strip()


# ── audio capture (parec -> raw s16le mono 16k on stdout) ─────────────────────
_buf = bytearray()
_lock = threading.Lock()


def _capture():
    p = subprocess.Popen(
        ["parec", "--format=s16le", f"--rate={RATE}", "--channels=1",
         "-d", SOURCE, "--latency-msec=100"],
        stdout=subprocess.PIPE)
    while True:
        chunk = p.stdout.read(3200)  # ~0.1s
        if not chunk:
            break
        with _lock:
            _buf.extend(chunk)


def _grab():
    with _lock:
        return bytes(_buf)


def _reset():
    with _lock:
        _buf.clear()


def _rms(a):
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def main():
    print(f"listening on: {SOURCE}\ncategory: {CATEGORY}\nplay/read a question…\n", flush=True)
    threading.Thread(target=_capture, daemon=True).start()

    client = httpx.Client(timeout=20)
    history, last_words, silent_cycles, buzzed = [], 0, 0, False

    SPEECH, QUIET = 0.004, 0.0035   # your mic is quiet; low thresholds
    while True:
        time.sleep(1.2)
        raw = _grab()
        if len(raw) < RATE * 2 * 0.6:            # < 0.6s captured
            continue
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms, tail_rms = _rms(audio), _rms(audio[-int(RATE * 0.8):])
        print(f"[dbg buf={len(audio)/RATE:4.1f}s rms={rms:.4f} tail={tail_rms:.4f}]", flush=True)

        # transcribe whenever the buffer holds real audio (don't drop on a pause)
        if rms > SPEECH:
            gain = min(10.0, 0.06 / max(rms, 0.004))   # boost quiet mic for clean STT
            text = _transcribe((audio * gain).astype(np.float32))
            nwords = len(text.split())
            if text and nwords > last_words:
                last_words = nwords
                try:
                    d = client.post(f"{LEBOT_URL}/analyze", json={
                        "prefix": text, "category": CATEGORY,
                        "total_words": max(40, nwords * 2), "history": history}).json()
                except Exception as e:
                    print("analyze error:", e, flush=True)
                    d = {}
                history.append({"guess": d.get("guess", ""), "mode": d.get("mode", "recall")})
                guess, p = d.get("guess", "?"), d.get("p_buzz", 0)
                print(f"[{nwords:2}w P={p:.2f} → {guess[:22]}] …{text[-45:]}", flush=True)
                if d.get("buzzes") and not buzzed:
                    buzzed = True
                    print(f"\n  ⚡⚡ BUZZ — {guess}  (P={p}, {nwords} words)\n", flush=True)
                    post_discord(f"⚡ **BUZZ** — **{guess}**  (P={p}, {nwords} words heard)\n> {text}")

        # sustained quiet after a question -> reset for the next one (patient: ~6s so
        # natural reading pauses don't chop a question in half)
        if tail_rms < QUIET:
            silent_cycles += 1
            if silent_cycles >= 5 and last_words > 0:
                print("  — silence, resetting for next question —\n", flush=True)
                _reset()
                history, last_words, silent_cycles, buzzed = [], 0, 0, False
        else:
            silent_cycles = 0
        if len(audio) > RATE * 40:               # cap buffer growth
            _reset()
            last_words = 0


if __name__ == "__main__":
    main()
