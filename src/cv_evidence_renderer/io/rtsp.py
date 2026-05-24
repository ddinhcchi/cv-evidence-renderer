"""Threaded RTSP reader with auto-reconnect. v0.2.

Pattern ported from
https://github.com/ddinhcchi/realtime-object-detection-alert/blob/main/src/rtsp.py
"""

from __future__ import annotations


class RtspReader:
    def __init__(self, url: str, transport: str = "tcp") -> None:
        self.url = url
        self.transport = transport

    def start(self) -> None:
        raise NotImplementedError("v0.2")

    def read(self) -> object:
        raise NotImplementedError("v0.2")

    def stop(self) -> None:
        raise NotImplementedError("v0.2")
