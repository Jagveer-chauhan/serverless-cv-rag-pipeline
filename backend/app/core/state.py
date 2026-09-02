"""Application runtime state tracking — including cold-start and inference timing."""
import time
from typing import Optional, Any


class AppState:
    def __init__(self):
        self.start_time: float = time.time()
        self.is_warm: bool = False
        self.embedding_model: Optional[Any] = None
        self.last_keepalive: float = time.time()

        # Cold-start / LLM warm-path tracking
        self.cold_start_ms: Optional[float] = None        # Time from boot to first inference
        self.first_inference_ms: Optional[float] = None   # Latency of the very first LLM call
        self.warm_inference_ms: Optional[float] = None    # Latency of subsequent warm calls
        self.llm_call_count: int = 0                      # Total LLM inference calls made
        self.cold_start_occurred: bool = False             # Whether a cold start was observed

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def record_inference(self, duration_ms: float) -> None:
        """Records an LLM inference call, distinguishing cold-start from warm-path."""
        self.llm_call_count += 1
        if self.llm_call_count == 1:
            # First ever inference — this IS the cold-start inference
            self.cold_start_occurred = True
            self.first_inference_ms = round(duration_ms, 2)
            self.cold_start_ms = round(self.uptime_seconds * 1000, 2)
        else:
            # Warm-path: rolling minimum of subsequent calls
            if self.warm_inference_ms is None or duration_ms < self.warm_inference_ms:
                self.warm_inference_ms = round(duration_ms, 2)


app_state = AppState()
