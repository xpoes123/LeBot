"""LeBot voice bot — sits in a Discord voice call, transcribes the reader with Whisper,
and streams the growing question text to the VPS anticipation pipeline, buzzing the moment
the buzzer commits.

Commands (prefix !):
  !join            join your current voice channel
  !go [category]   start a NEW question: reset, listen, transcribe, anticipate
  !stop            end the current question
  !leave           disconnect

The anticipation itself runs on the VPS (lebot.djiang.xyz /analyze) so no API key is
needed here — this process only does Discord audio + Whisper STT.
"""
import asyncio
import os
import threading
from pathlib import Path

import httpx
import numpy as np
import discord
from discord.ext import commands, voice_recv
from faster_whisper import WhisperModel


def _load_env():
    for line in Path(__file__).with_name(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
TOKEN = os.environ["DISCORD_TOKEN"]
LEBOT_URL = os.environ.get("LEBOT_URL", "https://lebot.djiang.xyz")
CATEGORIES = {"BIOLOGY", "CHEMISTRY", "PHYSICS", "EARTH_SPACE", "MATH", "ENERGY", "OTHER"}

print("loading Whisper…")
try:
    _model = WhisperModel("base.en", device="cuda", compute_type="float16")
    print("Whisper on GPU (base.en, float16)")
except Exception as e:
    print(f"GPU unavailable ({e}); falling back to CPU")
    _model = WhisperModel("base.en", device="cpu", compute_type="int8")


def _transcribe(audio):
    segs, _ = _model.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text for s in segs).strip()


class Session:
    """One question's rolling audio buffer + state."""
    def __init__(self):
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.active = False
        self.buzzed = False
        self.category = "OTHER"
        self.last_words = 0

    def add_pcm(self, pcm):
        with self.lock:
            self.buf.extend(pcm)

    def audio16k(self):
        """48 kHz stereo s16le (Discord) -> 16 kHz mono float32 (Whisper)."""
        with self.lock:
            raw = bytes(self.buf)
        if len(raw) < 4:
            return None
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        a = a.reshape(-1, 2).mean(axis=1)   # stereo -> mono
        a = a[::3]                          # 48k -> 16k (exact 3:1). ponytail: crude decimate,
        return a / 32768.0                  # add a low-pass if aliasing hurts accuracy


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
sessions = {}          # guild_id -> Session


def _sink(session):
    def cb(user, data: voice_recv.VoiceData):
        if user and getattr(user, "bot", False):
            return
        session.add_pcm(data.pcm)
    return voice_recv.BasicSink(cb)


async def _loop(session, channel):
    """Every ~1s: transcribe the whole question so far, and if new words landed, ask the
    VPS whether to buzz. Posts a throttled live line + a big BUZZ when it commits."""
    loop = asyncio.get_event_loop()
    async with httpx.AsyncClient(timeout=20) as client:
        while session.active:
            await asyncio.sleep(1.0)
            audio = session.audio16k()
            if audio is None or len(audio) < 16000 * 0.6:   # < 0.6s of audio
                continue
            text = await loop.run_in_executor(None, _transcribe, audio)
            nwords = len(text.split())
            if nwords <= session.last_words or nwords == 0:
                continue
            session.last_words = nwords
            try:
                r = await client.post(f"{LEBOT_URL}/analyze", json={
                    "prefix": text, "category": session.category,
                    "total_words": max(40, nwords * 2)})
                d = r.json()
            except Exception as e:
                print("analyze error:", e)
                continue
            guess, p = d.get("guess", "?"), d.get("p_buzz", 0)
            print(f"[{nwords}w P={p}] {text[-60:]!r} -> {guess}")
            if d.get("buzzes") and not session.buzzed:
                session.buzzed = True
                await channel.send(f"⚡ **BUZZ** — **{guess}**   (P={p}, {nwords} words heard)\n> {text}")


@bot.event
async def on_ready():
    print(f"logged in as {bot.user}")


@bot.command()
async def join(ctx):
    if not ctx.author.voice:
        await ctx.send("Join a voice channel first, then `!join`.")
        return
    ch = ctx.author.voice.channel
    if ctx.voice_client:
        await ctx.voice_client.move_to(ch)
    else:
        await ch.connect(cls=voice_recv.VoiceRecvClient)
    sessions[ctx.guild.id] = Session()
    await ctx.send(f"In **{ch.name}**. `!go [category]` to start a question.")


@bot.command()
async def go(ctx, category: str = "OTHER"):
    vc = ctx.voice_client
    if not vc:
        await ctx.send("`!join` first.")
        return
    s = sessions.setdefault(ctx.guild.id, Session())
    s.__init__()
    s.category = category.upper() if category.upper() in CATEGORIES else "OTHER"
    s.active = True
    if vc.is_listening():
        vc.stop_listening()
    vc.listen(_sink(s))
    asyncio.create_task(_loop(s, ctx.channel))
    await ctx.send(f"🎧 Listening ({s.category}). Read the question…")


@bot.command()
async def stop(ctx):
    s = sessions.get(ctx.guild.id)
    if s:
        s.active = False
    if ctx.voice_client and ctx.voice_client.is_listening():
        ctx.voice_client.stop_listening()
    await ctx.send("Stopped." + ("" if not s or s.buzzed else " (never buzzed)"))


@bot.command()
async def leave(ctx):
    s = sessions.get(ctx.guild.id)
    if s:
        s.active = False
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
    await ctx.send("👋")


if __name__ == "__main__":
    bot.run(TOKEN)
