"""
ROI Calibrator — Interactive helper to find the washing-machine panel ROI.

Opens the calibration image (assets/teste_frame.jpg), lets you draw a
rectangle around the machine's display, and prints the exact config
snippet to paste into config.yaml.

Usage:
    python roi_calibrator.py
    python roi_calibrator.py --image assets/teste_frame.jpg
    python roi_calibrator.py --width 960 --height 540
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml


DEFAULT_IMAGE = Path("assets/teste_frame.jpg")
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


def calibrate(
    image_path: Path,
    display_width: int = DEFAULT_WIDTH,
    display_height: int = DEFAULT_HEIGHT,
) -> None:
    """
    Open an interactive OpenCV window to select the panel ROI.

    Args:
        image_path: Path to the calibration frame.
        display_width: Width to resize the image for display (and monitoring).
        display_height: Height to resize the image for display.
    """
    if not image_path.exists():
        print(f"[ERRO] Imagem não encontrada: {image_path}", file=sys.stderr)
        print("       Verifique o caminho ou capture um novo frame da câmera.", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERRO] Falha ao decodificar a imagem: {image_path}", file=sys.stderr)
        sys.exit(1)

    resized = cv2.resize(img, (display_width, display_height))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    print("┌─────────────────────────────────────────────────────────┐")
    print("│  ROI Calibrator                                         │")
    print("│  Desenhe um retângulo em volta do painel iluminado.     │")
    print("│  Pressione ENTER para confirmar ou ESC para cancelar.   │")
    print("└─────────────────────────────────────────────────────────┘")

    bbox = cv2.selectROI("Selecione o painel — ENTER confirma, ESC cancela", gray, False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, bbox)

    if w == 0 or h == 0:
        print("\n[Nenhuma seleção feita — saindo sem alterações.]")
        sys.exit(0)

    y_start, y_end = y, y + h
    x_start, x_end = x, x + w

    print("\n" + "=" * 60)
    print("  COLE ESTE BLOCO NO SEU config.yaml  (detection.roi):")
    print("=" * 60)
    print(f"  roi:")
    print(f"    y_start: {y_start}")
    print(f"    y_end: {y_end}")
    print(f"    x_start: {x_start}")
    print(f"    x_end: {x_end}")
    print("=" * 60)
    print(f"\n  Slice equivalente (debug): frame[{y_start}:{y_end}, {x_start}:{x_end}]")
    print(f"  Tamanho do ROI: {w} x {h} px  (em {display_width}x{display_height})\n")

    # Optionally offer to patch config.yaml in-place
    config_path = Path("config.yaml")
    if config_path.exists():
        answer = input("  Atualizar config.yaml automaticamente? [s/N]: ").strip().lower()
        if answer in ("s", "sim", "y", "yes"):
            _patch_config(config_path, y_start, y_end, x_start, x_end)
            print(f"  ✔ config.yaml atualizado!")


def _patch_config(path: Path, y_start: int, y_end: int, x_start: int, x_end: int) -> None:
    """Patch only the ROI keys in an existing config.yaml, preserving other values."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["detection"]["roi"]["y_start"] = y_start
    data["detection"]["roi"]["y_end"] = y_end
    data["detection"]["roi"]["x_start"] = x_start
    data["detection"]["roi"]["x_end"] = x_end
    # Preserve formatting reasonably — yaml.dump with sort_keys=False
    path.write_text(
        yaml.dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the panel ROI")
    parser.add_argument(
        "--image", "-i", type=Path, default=DEFAULT_IMAGE,
        help=f"Path to calibration image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Display width (default: 960)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Display height (default: 540)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    calibrate(args.image, args.width, args.height)
