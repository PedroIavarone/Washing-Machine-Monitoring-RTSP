"""
Washing Machine Monitoring — RTSP Panel Monitor

Captures an RTSP stream via FFmpeg, analyses the brightness of a
configurable ROI, and triggers a cross-platform audio alarm when the
washing-machine panel goes dark (machine finished).

Usage:
    python monitor.py                  # uses config.yaml
    python monitor.py --config my.yaml # custom config path
"""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Config dataclasses

@dataclass(frozen=True)
class ROI:
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def slice_str(self) -> str:
        return f"[{self.y_start}:{self.y_end}, {self.x_start}:{self.x_end}]"


@dataclass(frozen=True)
class DetectionConfig:
    roi: ROI
    threshold: int
    pixel_limit: int
    frame_skip: int
    consecutive_off: int
    display_width: int
    display_height: int


@dataclass(frozen=True)
class AlarmConfig:
    audio_path: Path
    loop: bool


@dataclass(frozen=True)
class StreamConfig:
    reconnect_attempts: int
    reconnect_delay_sec: float
    ffmpeg_timeout_sec: int


@dataclass(frozen=True)
class RTSPConfig:
    url: str
    width: int
    height: int


@dataclass(frozen=True)
class AppConfig:
    rtsp: RTSPConfig
    detection: DetectionConfig
    alarm: AlarmConfig
    stream: StreamConfig

    @staticmethod
    def from_yaml(path: Path) -> "AppConfig":
        """Load and validate configuration from a YAML file."""
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            roi_raw = raw["detection"]["roi"]
            return AppConfig(
                rtsp=RTSPConfig(
                    url=raw["rtsp"]["url"],
                    width=int(raw["rtsp"]["width"]),
                    height=int(raw["rtsp"]["height"]),
                ),
                detection=DetectionConfig(
                    roi=ROI(
                        y_start=int(roi_raw["y_start"]),
                        y_end=int(roi_raw["y_end"]),
                        x_start=int(roi_raw["x_start"]),
                        x_end=int(roi_raw["x_end"]),
                    ),
                    threshold=int(raw["detection"]["threshold"]),
                    pixel_limit=int(raw["detection"]["pixel_limit"]),
                    frame_skip=int(raw["detection"]["frame_skip"]),
                    consecutive_off=int(raw["detection"]["consecutive_off"]),
                    display_width=int(raw["detection"]["display_width"]),
                    display_height=int(raw["detection"]["display_height"]),
                ),
                alarm=AlarmConfig(
                    audio_path=Path(raw["alarm"]["audio_path"]),
                    loop=bool(raw["alarm"]["loop"]),
                ),
                stream=StreamConfig(
                    reconnect_attempts=int(raw["stream"]["reconnect_attempts"]),
                    reconnect_delay_sec=float(raw["stream"]["reconnect_delay_sec"]),
                    ffmpeg_timeout_sec=int(raw["stream"]["ffmpeg_timeout_sec"]),
                ),
            )
        except KeyError as exc:
            raise ValueError(f"Missing required config key: {exc}") from exc



# Image processing

class PanelDetector:
    """Detects whether the washing-machine panel is ON or OFF by brightness."""

    def __init__(self, config: DetectionConfig) -> None:
        self._cfg = config
        self._off_count: int = 0

    def is_panel_off(self, display_frame: np.ndarray) -> tuple[bool, int]:
        """
        Analyse the ROI of a display frame.

        Returns:
            (is_off, white_pixel_count)
        """
        roi = display_frame[
            self._cfg.roi.y_start : self._cfg.roi.y_end,
            self._cfg.roi.x_start : self._cfg.roi.x_end,
        ]
        _, mask = cv2.threshold(roi, self._cfg.threshold, 255, cv2.THRESH_BINARY)
        white_pixels = int(cv2.countNonZero(mask))
        is_off = white_pixels < self._cfg.pixel_limit
        return is_off, white_pixels

    def update(self, is_off: bool) -> bool:
        """
        Feed a new reading and return True if the alarm should fire.

        Uses a consecutive-reading debounce to avoid false positives from
        momentary reflections or frame glitches.
        """
        if is_off:
            self._off_count += 1
        else:
            self._off_count = 0
        return self._off_count >= self._cfg.consecutive_off

    def get_mask(self, display_frame: np.ndarray) -> np.ndarray:
        """Return the binary mask of the ROI (useful for debug window)."""
        roi = display_frame[
            self._cfg.roi.y_start : self._cfg.roi.y_end,
            self._cfg.roi.x_start : self._cfg.roi.x_end,
        ]
        _, mask = cv2.threshold(roi, self._cfg.threshold, 255, cv2.THRESH_BINARY)
        return mask

    @property
    def off_count(self) -> int:
        return self._off_count



# Cross-platform alarm

class AlarmPlayer:
    """
    Plays an audio file in a loop on a background thread.

    Strategy (best-effort, platform-aware):
      1. Try pygame.mixer (cross-platform, preferred).
      2. Fallback to a platform-native CLI player via subprocess.
    """

    def __init__(self, audio_path: Path, loop: bool = True) -> None:
        self._audio_path = audio_path
        self._loop = loop
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playing = False
        self._lock = threading.Lock()

    # public API

    def start(self) -> None:
        """Start the alarm (no-op if already playing)."""
        with self._lock:
            if self._playing:
                return
            if not self._audio_path.exists():
                log.warning("Alarm audio not found: %s — skipping playback.", self._audio_path)
                return
            self._stop_event.clear()
            self._playing = True

        if self._try_pygame():
            return
        self._start_subprocess_loop()

    def stop(self) -> None:
        """Stop the alarm if it is playing."""
        with self._lock:
            if not self._playing:
                return
            self._playing = False
        self._stop_event.set()
        self._stop_subprocess()
        self._stop_pygame()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    @property
    def is_playing(self) -> bool:
        return self._playing

    # pygame backend

    def _try_pygame(self) -> bool:
        try:
            import pygame  # type: ignore[import-untyped]

            pygame.mixer.init()
            pygame.mixer.music.load(str(self._audio_path))
            pygame.mixer.music.play(loops=-1 if self._loop else 0)
            log.info("Alarm playing via pygame.mixer (loop=%s).", self._loop)
            return True
        except ImportError:
            log.info("pygame not installed — falling back to subprocess audio.")
            return False
        except Exception as exc:
            log.warning("pygame playback failed (%s) — falling back to subprocess.", exc)
            return False

    def _stop_pygame(self) -> None:
        try:
            import pygame  # type: ignore[import-untyped]

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception:
            pass

    # subprocess fallback

    def _start_subprocess_loop(self) -> None:
        def _loop() -> None:
            player = self._native_player_cmd()
            while not self._stop_event.is_set():
                try:
                    self._process = subprocess.Popen(
                        player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    # Poll until the clip ends or stop is requested
                    while self._process.poll() is None and not self._stop_event.is_set():
                        time.sleep(0.2)
                    if self._process.poll() is None:
                        self._process.terminate()
                        self._process.wait(timeout=3)
                    if not self._loop:
                        break
                except Exception as exc:
                    log.error("Subprocess audio error: %s", exc)
                    break

        self._thread = threading.Thread(target=_loop, daemon=True, name="alarm-audio")
        self._thread.start()
        log.info("Alarm playing via subprocess (%s, loop=%s).", self._native_player_cmd()[0], self._loop)

    def _stop_subprocess(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        # macOS afplay: make sure no orphaned afplay lingers
        if platform.system() == "Darwin":
            subprocess.run(["killall", "afplay"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _native_player_cmd_static(audio: Path) -> list[str]:
        system = platform.system()
        if system == "Darwin":
            return ["afplay", str(audio)]
        if system == "Linux":
            # Try common Linux players in order
            for cmd in ("mpg123", "mpg321", "ffplay", "aplay"):
                from shutil import which

                if which(cmd):
                    if cmd == "ffplay":
                        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio)]
                    return [cmd, str(audio)]
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio)]
        if system == "Windows":
            # PowerShell Media.SoundPlayer or fallback
            return [
                "powershell",
                "-c",
                f'(New-Object Media.SoundPlayer "{audio}").PlaySync()',
            ]
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio)]

    def _native_player_cmd(self) -> list[str]:
        return self._native_player_cmd_static(self._audio_path)



# FFmpeg RTSP capture

class RTSPCapture:
    """Wraps an FFmpeg subprocess that yields raw BGR frames."""

    def __init__(self, config: RTSPConfig, ffmpeg_timeout: int = 10) -> None:
        self._cfg = config
        self._timeout = ffmpeg_timeout
        self._process: Optional[subprocess.Popen] = None
        self._frame_size = config.width * config.height * 3

    def open(self) -> None:
        """Start (or restart) the FFmpeg process."""
        self.close()
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-stimeout", str(self._timeout * 1_000_000),  # microseconds
            "-i", self._cfg.url,
            "-an",
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-",
        ]
        log.info("Starting FFmpeg capture: %sx%s", self._cfg.width, self._cfg.height)
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self) -> Optional[np.ndarray]:
        """
        Read one raw frame. Returns None on EOF / error.
        """
        if self._process is None or self._process.stdout is None:
            return None
        raw = self._process.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape([self._cfg.height, self._cfg.width, 3])

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def close(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None



# Main loop

def run_monitor(config: AppConfig) -> None:
    """Main monitoring loop with automatic reconnection."""
    capture = RTSPCapture(config.rtsp, ffmpeg_timeout=config.stream.ffmpeg_timeout_sec)
    detector = PanelDetector(config.detection)
    alarm = AlarmPlayer(config.alarm.audio_path, loop=config.alarm.loop)

    frame_count = 0
    reconnects = 0

    log.info("ROI=%s  threshold=%d  pixel_limit=%d  frame_skip=%d",
             config.detection.roi.slice_str(),
             config.detection.threshold,
             config.detection.pixel_limit,
             config.detection.frame_skip)
    log.info("Press 'q' to quit. Panel debug window shows the binary mask.")

    try:
        capture.open()

        while True:
            frame = capture.read()

            # reconnection logic
            if frame is None:
                if not capture.is_alive():
                    reconnects += 1
                    if reconnects > config.stream.reconnect_attempts:
                        log.error("Max reconnection attempts (%d) reached — exiting.",
                                  config.stream.reconnect_attempts)
                        break
                    log.warning("Stream lost — reconnecting (%d/%d) in %ds...",
                                reconnects, config.stream.reconnect_attempts,
                                config.stream.reconnect_delay_sec)
                    time.sleep(config.stream.reconnect_delay_sec)
                    capture.open()
                    continue
                # Short read but process still alive — skip
                continue

            reconnects = 0  # successful read resets counter

            # frame processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = cv2.resize(gray, (config.detection.display_width, config.detection.display_height))

            if frame_count % config.detection.frame_skip == 0:
                is_off, white_px = detector.is_panel_off(display)
                should_alarm = detector.update(is_off)

                log.info("Pixels=%4d  off_count=%d  panel=%s",
                         white_px, detector.off_count, "OFF" if is_off else "ON ")

                if should_alarm and not alarm.is_playing:
                    log.warning("Panel OFF for %d consecutive readings — triggering alarm!",
                                config.detection.consecutive_off)
                    alarm.start()

                # Debug: show binary mask of the ROI
                try:
                    mask = detector.get_mask(display)
                    cv2.imshow("Panel ROI (binary mask)", mask)
                except cv2.error:
                    pass  # headless environment

            frame_count += 1

            try:
                cv2.imshow("Washing Machine Monitor", display)
            except cv2.error:
                pass  # headless

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit requested (q).")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")

    finally:
        log.info("Shutting down...")
        alarm.stop()
        capture.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass



# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Washing Machine RTSP Monitor")
    parser.add_argument(
        "--config", "-c", type=Path, default=Path("config.yaml"),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        log.error("Copy config.yaml.example or create config.yaml first.")
        sys.exit(1)

    try:
        config = AppConfig.from_yaml(args.config)
    except (ValueError, yaml.YAMLError) as exc:
        log.error("Failed to load config: %s", exc)
        sys.exit(1)

    run_monitor(config)


if __name__ == "__main__":
    main()
