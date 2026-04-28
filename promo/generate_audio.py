"""Generate narration audio using Google Text-to-Speech."""
from gtts import gTTS

NARRATION = (
    "Introducing eStocks. Trading scripts and plugins for the modern investor. Feature one: Real-time market data feeds from multiple exchanges. Feature two: Algorithmic trading strategies with backtesting. Feature three: Portfolio analytics with risk management and reporting. eStocks. Open source and data driven. Visit github dot com slash embeddedos-org slash eStocks."
)

tts = gTTS(text=NARRATION, lang="en", slow=False)
tts.save("narration.mp3")
print(f"Generated narration.mp3 ({len(NARRATION)} chars)")
