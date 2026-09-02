"""Application runtime state tracking."""
import time
from typing import Optional, Any


class AppState:
    def __init__(self):
        self.start_time: float = time.time()
        self.is_warm: bool = False
        self.embedding_model: Optional[Any] = None
        self.last_keepalive: float = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time


app_state = AppState()
