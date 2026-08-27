"""Starting the local model server ourselves, so its load time is free.

A model server is slow to start for exactly one reason: it reads gigabytes off
disk before it will answer. Waiting for that as a *step* — start the server,
watch a progress bar, time out, try again — is the worst possible arrangement,
because the wait is unavoidable and the person is doing nothing during it.

So the program launches it at startup and carries on. The model loads in the
background while you are reading the greeting and typing what you want built,
and by the time the first instruction is finished the server is usually ready.
If it is not, the first build waits — once, with a message — instead of failing.

Nothing here is required: if there is no server binary or no model file, this
finds nothing and the program behaves exactly as it did before.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voicevibecoder.codegen.local_brain import port_open
from voicevibecoder.config import Config

# Where a model file plausibly lives, in the order worth looking.
MODEL_DIRS = ("~/models", "~", "~/storage/downloads", "~/downloads")
BINARIES = ("llama-server", "llama-cpp-server")


@dataclass
class ModelServer:
    """A llama.cpp server this program is responsible for."""

    binary: str
    model_path: Path
    port: int
    log_path: Path
    ctx: int = 4096
    process: subprocess.Popen | None = None

    # -- discovery -------------------------------------------------------
    @classmethod
    def discover(cls, config: Config) -> ModelServer | None:
        """A server we could start, or None if the pieces are not here."""
        binary = next((found for name in BINARIES if (found := shutil.which(name))), None)
        if binary is None:
            return None
        model = find_model(config)
        if model is None:
            return None
        return cls(
            binary=binary,
            model_path=model,
            port=port_of(config.local_url),
            log_path=model.parent / "llama-server.log",
            ctx=config.local_ctx,
        )

    # -- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        """Launch it in the background. False if the port is already taken."""
        if port_open(self.port):
            return False  # something owns this port; never race it
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("wb") as log:
            self.process = subprocess.Popen(  # noqa: S603 — argv from a found binary
                [
                    self.binary,
                    "-m",
                    str(self.model_path),
                    "--port",
                    str(self.port),
                    "--ctx-size",
                    str(self.ctx),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # survives this program closing
            )
        return True

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def ready(self) -> bool:
        return port_open(self.port)

    def wait(
        self,
        timeout_s: float = 600,
        on_tick: Callable[[float], None] | None = None,
        interval_s: float = 2.0,
    ) -> bool:
        """Block until it answers, reporting progress. False on timeout/death."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready():
                return True
            if self.process is not None and self.process.poll() is not None:
                return False  # it died; the log says why
            if on_tick:
                on_tick(time.monotonic() - (deadline - timeout_s))
            time.sleep(interval_s)
        return self.ready()

    def tail(self, lines: int = 12) -> str:
        try:
            return "\n".join(
                self.log_path.read_text("utf-8", "replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def describe(self) -> str:
        return f"{self.model_path.name} on port {self.port}"


def find_model(config: Config) -> Path | None:
    """The GGUF to load: the configured one, else the largest one lying around.

    Largest, because a directory with several usually has one real model and a
    couple of small experiments, and the real one is what was downloaded on
    purpose.
    """
    if config.local_model_file:
        named = Path(config.local_model_file).expanduser()
        return named if named.is_file() else None

    found: list[Path] = []
    for directory in MODEL_DIRS:
        root = Path(directory).expanduser()
        if root.is_dir():
            found.extend(path for path in root.glob("*.gguf") if path.is_file())
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_size)


def port_of(url: str, default: int = 11434) -> int:
    from urllib.parse import urlsplit  # noqa: PLC0415

    return urlsplit(url.rstrip("/")).port or default


def is_termux() -> bool:
    return bool(os.environ.get("TERMUX_VERSION")) or "com.termux" in os.environ.get(
        "PREFIX", ""
    )
