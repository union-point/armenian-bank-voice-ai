import logging
import time

from livekit import agents
from livekit.agents import AgentServer, AgentSession
from livekit.plugins import  google, silero

from src.agent.assistant import BankAssistant
from src.agent.config import config
from src.audio import FasterWhisperSTT
from livekit.plugins import groq
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

server = AgentServer()


@server.rtc_session(agent_name="bank-assistant")
async def entrypoint(ctx: agents.JobContext):
    logger.info("Initializing AgentSession...")

    session = AgentSession(
        # stt=FasterWhisperSTT(
        #     model_name_or_path=config.stt_model_path,
        #     device=config.stt_device,
        #     compute_type=config.stt_compute_type,
        #     language="hy",
        # ),
        stt=groq.STT(
      model="whisper-large-v3",
      language="hy",
   ),
        llm=google.LLM(model="gemini-2.5-flash"),
       tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Kore",
            instructions="Խոսիր քաղաքավարի և բնական հայերենով",
        ),
        vad=silero.VAD.load(),
    )

    # Measures time from user speech transcribed to agent starting to speak.
    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        logger.debug(f"User said: {ev.transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

    await session.start(
        room=ctx.room,
        agent=BankAssistant(),
    )

    await session.generate_reply(
        instructions=(
            "Բարևիր օգտվողին և ասա, որ կարող ես օգնել միայն վարկերի, ավանդների "
            "և մասնաճյուղերի  թեմաներով՝ բանկերի պաշտոնական կայքերի տվյալներով։"
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
