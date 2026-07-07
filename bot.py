"""LeBot control bot — one local process you drive from Discord.

Connects to Discord for slash commands (gateway only — no voice receive, which is broken
against Discord's current protocol), captures audio locally with parec (your mic OR your
computer's output), transcribes on the GPU, runs the anticipation pipeline, and posts
buzzes to the channel.

Slash commands:
  /listen source:<mic|computer> [category]   start listening
  /stop                                        stop
  /status                                      what it's doing

Runs LOCALLY only. Holds no Claude key — anticipation is a POST to the VPS.
"""
import glob
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import discord
from faster_whisper import WhisperModel


def _load_env():
    for line in Path(__file__).with_name(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
TOKEN = os.environ["DISCORD_TOKEN"]
LEBOT_URL = os.environ.get("LEBOT_URL", "https://lebot.djiang.xyz")
RATE = 16000
CATS = ["BIOLOGY", "CHEMISTRY", "PHYSICS", "EARTH_SPACE", "MATH", "ENERGY", "OTHER"]


def _load_whisper():
    import ctypes
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


print("loading Whisper…", flush=True)
_model = _load_whisper()


def _transcribe(audio):
    segs, _ = _model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text for s in segs).strip()


def _mic_source():
    return subprocess.run(["pactl", "get-default-source"], capture_output=True, text=True).stdout.strip()


def _computer_source():
    sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True).stdout.strip()
    return f"{sink}.monitor"


def _rms(a):
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def _post(channel_id, msg):
    try:
        httpx.post(f"https://discord.com/api/v10/channels/{channel_id}/messages",
                   headers={"Authorization": f"Bot {TOKEN}"},
                   json={"content": msg[:1900]}, timeout=10)
    except Exception as e:
        print("post err:", e, flush=True)


class Listener:
    """Captures one audio source, transcribes, anticipates, buzzes — until stopped."""
    def __init__(self):
        self._stop = threading.Event()
        self.thread = None
        self.status = "idle"

    def start(self, source, category, channel_id):
        self.stop()
        self._stop.clear()
        self.status = f"listening ({category}) on {source.split('.')[0][-24:]}"
        self.thread = threading.Thread(
            target=self._run, args=(source, category, channel_id), daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.status = "idle"

    def _run(self, source, category, channel_id):
        buf = bytearray()
        lock = threading.Lock()

        def reader():
            p = subprocess.Popen(
                ["parec", "--format=s16le", f"--rate={RATE}", "--channels=1",
                 "-d", source, "--latency-msec=100"], stdout=subprocess.PIPE)
            self._parec = p
            while not self._stop.is_set():
                chunk = p.stdout.read(3200)
                if not chunk:
                    break
                with lock:
                    buf.extend(chunk)
            p.terminate()

        threading.Thread(target=reader, daemon=True).start()

        client = httpx.Client(timeout=20)
        history, last_words, stale, buzzed = [], 0, 0, False
        best_guess, best_p = "", 0.0
        SPEECH = 0.004
        while not self._stop.is_set():
            time.sleep(0.8)
            with lock:
                raw = bytes(buf)
            if len(raw) < RATE * 2 * 0.6:
                continue
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            grew = False
            if _rms(audio) > SPEECH:
                gain = min(10.0, 0.06 / max(_rms(audio), 0.004))
                t0 = time.time()
                text = _transcribe((audio * gain).astype(np.float32))
                t_stt = time.time() - t0
                nwords = len(text.split())
                if text and nwords > last_words:
                    grew, last_words = True, nwords
                    t1 = time.time()
                    try:
                        d = client.post(f"{LEBOT_URL}/analyze", json={
                            "prefix": text, "category": category,
                            "total_words": max(40, nwords * 2), "history": history, "fast": True}).json()
                    except Exception:
                        d = {}
                    print(f"[timing buf={len(audio)/RATE:.0f}s stt={t_stt:.1f}s analyze={time.time()-t1:.1f}s]", flush=True)
                    history.append({"guess": d.get("guess", ""), "mode": d.get("mode", "recall")})
                    guess, p = d.get("guess", "?"), d.get("p_buzz", 0)
                    if guess and guess.upper() != "UNKNOWN":
                        best_guess, best_p = guess, p
                    print(f"[{nwords:2}w P={p:.2f} → {guess[:22]}] …{text[-45:]}", flush=True)
                    if d.get("buzzes") and not buzzed:
                        buzzed = True
                        _post(channel_id, f"⚡ **BUZZ** — **{guess}**  (P={p}, {nwords} words)\n> {text}")

            # end of question = transcript stopped growing for ~5s (volume-independent)
            if grew:
                stale = 0
            elif last_words > 0:
                stale += 1
                if stale >= 5:
                    if not buzzed and best_guess:
                        _post(channel_id, f"🔔 **(end of question) best guess** — **{best_guess}**  (P={best_p})")
                    print("— reset —", flush=True)
                    with lock:
                        buf.clear()
                    history, last_words, stale, buzzed = [], 0, 0, False
                    best_guess, best_p = "", 0.0
            if len(audio) > RATE * 45:
                with lock:
                    buf.clear()
                last_words = 0


listener = Listener()
bot = discord.Bot(intents=discord.Intents.default())


@bot.event
async def on_ready():
    await bot.sync_commands(guild_ids=[g.id for g in bot.guilds], force=True)
    print(f"logged in as {bot.user} — ready", flush=True)


@bot.slash_command(name="listen", description="Start listening (mic = you read, computer = Discord call audio)")
async def listen_cmd(
    ctx: discord.ApplicationContext,
    source: discord.Option(str, "Audio source", choices=["mic", "computer"]) = "mic",
    category: discord.Option(str, "Question category", choices=CATS) = "OTHER",
):
    src = _mic_source() if source == "mic" else _computer_source()
    listener.start(src, category.upper(), ctx.channel_id)
    await ctx.respond(f"🎧 Listening to **{source}** ({category}). I'll buzz here. `/stop` to end.")


@bot.slash_command(name="stop", description="Stop listening")
async def stop_cmd(ctx: discord.ApplicationContext):
    listener.stop()
    await ctx.respond("⏹️ Stopped.")


@bot.slash_command(name="status", description="What is LeBot doing?")
async def status_cmd(ctx: discord.ApplicationContext):
    await ctx.respond(f"Status: {listener.status}")


if __name__ == "__main__":
    bot.run(TOKEN)
