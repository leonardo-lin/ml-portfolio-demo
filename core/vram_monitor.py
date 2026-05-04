"""
Background thread VRAM + system RAM monitor using pynvml + psutil.
Thread-safe, non-blocking for Streamlit.
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional


class VRAMMonitor:
    def __init__(self, poll_interval_sec: float = 0.5, history_len: int = 120):
        self._interval = poll_interval_sec
        self._history: deque = deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._handle = None
        self._available = False
        self._total_mb = 4096
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._total_mb = info.total // (1024 ** 2)
            self._available = True
        except Exception:
            self._available = False

    def _poll(self):
        try:
            import psutil
            import pynvml
            while not self._stop_event.is_set():
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    mem = psutil.virtual_memory()
                    record = {
                        "used_mb": info.used // (1024 ** 2),
                        "free_mb": info.free // (1024 ** 2),
                        "total_mb": info.total // (1024 ** 2),
                        "pct": round(info.used / info.total * 100, 1),
                        "ram_used_mb": mem.used // (1024 ** 2),
                        "ram_total_mb": mem.total // (1024 ** 2),
                        "ram_pct": round(mem.percent, 1),
                        "ts": time.time(),
                    }
                    with self._lock:
                        self._history.append(record)
                except Exception:
                    pass
                time.sleep(self._interval)
        except Exception:
            pass

    def start(self):
        if not self._available or (self._thread and self._thread.is_alive()):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_current(self) -> Dict:
        import psutil
        mem = psutil.virtual_memory()
        ram_fallback = {
            "ram_used_mb": mem.used // (1024 ** 2),
            "ram_total_mb": mem.total // (1024 ** 2),
            "ram_pct": round(mem.percent, 1),
        }
        if not self._available:
            return {"used_mb": 0, "free_mb": self._total_mb, "total_mb": self._total_mb, "pct": 0.0, **ram_fallback}
        with self._lock:
            if self._history:
                return dict(self._history[-1])
        # fallback: direct read
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "used_mb": info.used // (1024 ** 2),
                "free_mb": info.free // (1024 ** 2),
                "total_mb": info.total // (1024 ** 2),
                "pct": round(info.used / info.total * 100, 1),
                **ram_fallback,
            }
        except Exception:
            return {"used_mb": 0, "free_mb": self._total_mb, "total_mb": self._total_mb, "pct": 0.0, **ram_fallback}

    def get_history(self) -> List[Dict]:
        with self._lock:
            return list(self._history)

    @property
    def total_mb(self) -> int:
        return self._total_mb

    @property
    def is_available(self) -> bool:
        return self._available
