"""Generate per-segment narration using edge-tts (US English neural voice)."""
import asyncio
import json
import edge_tts
from mutagen.mp3 import MP3

# en-US-GuyNeural = neutral male US voice (Silicon Valley style)
VOICE = "en-US-GuyNeural"
RATE = "+0%"  # natural pace

SEGMENTS = [
    {"id": "intro", "text": "Introducing eStocks. Trading scripts and plugins for the modern investor."},
    {"id": "f1", "text": "Feature one. Real-Time Market Data. WebSocket feeds from NYSE, NASDAQ, and crypto exchanges with sub-second latency."},
    {"id": "f2", "text": "Feature two. Algorithmic Strategies. Backtestable strategy framework with moving averages, RSI, and custom indicators."},
    {"id": "f3", "text": "Feature three. Portfolio Analytics. Risk metrics, Sharpe ratio, drawdown analysis, and automated rebalancing."},
    {"id": "arch", "text": "Under the hood, eStocks is built with Python, Pandas, and WebSocket. The architecture flows from Data Feed, to Strategy Engine, to Risk Manager, to Order Router, to Analytics."},
    {"id": "cta", "text": "eStocks. Open source and data driven. Visit github dot com slash embeddedos-org slash eStocks."}
]


async def generate():
    durations = {}
    audio_files = []

    for seg in SEGMENTS:
        filename = f"seg_{seg['id']}.mp3"
        communicate = edge_tts.Communicate(seg["text"], VOICE, rate=RATE)
        await communicate.save(filename)
        dur = MP3(filename).info.length
        durations[seg["id"]] = round(dur + 0.5, 1)
        audio_files.append(filename)
        print(f"  {seg['id']}: {dur:.1f}s -> padded {durations[seg['id']]}s")

    with open("durations.json", "w") as f:
        json.dump(durations, f, indent=2)

    # Concatenate
    import subprocess
    with open("concat_list.txt", "w") as f:
        for af in audio_files:
            f.write(f"file '{af}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt", "-c", "copy", "narration.mp3"
    ], check=True)

    total = sum(durations.values())
    print(f"\nVoice: {VOICE}")
    print(f"Total narration: {total:.1f}s")
    print(f"Durations: {json.dumps(durations)}")


asyncio.run(generate())
