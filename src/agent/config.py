import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    # LiveKit settings
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")

    # AI external APIs
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    eleven_api_key: str = os.getenv("ELEVEN_API_KEY", "")

    # Local STT
    stt_model_path: str = os.getenv("STT_MODEL_PATH", "whisper-small-hy-ct2")
    stt_device: str = os.getenv("STT_DEVICE", "cpu")
    stt_chunk_size: float = float(os.getenv("STT_CHUNK_SIZE", "1.0"))
    stt_compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8")


config = AgentConfig()
