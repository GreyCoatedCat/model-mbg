"""
Export model hasil training ke format yang cepat di CPU Raspberry Pi 4.

NCNN adalah format yang direkomendasikan Ultralytics untuk Raspberry Pi
(dioptimasi untuk CPU ARM/NEON). ONNX disediakan sebagai pembanding.
Jalankan di laptop, lalu salin folder hasil export ke Raspi.

Contoh:
    python export_model.py --weights runs/mbg/seg-n/weights/best.pt
    python export_model.py --weights runs/mbg/seg-n/weights/best.pt --imgsz 480 --format onnx
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path best.pt")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="ukuran input di Raspi; lebih kecil = lebih cepat, akurasi turun")
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx"])
    ap.add_argument("--half", action="store_true",
                    help="FP16 (hanya NCNN); bisa sedikit lebih cepat, cek akurasinya")
    args = ap.parse_args()

    model = YOLO(args.weights)
    kwargs = dict(format=args.format, imgsz=args.imgsz)
    if args.format == "ncnn":
        kwargs["half"] = args.half
    else:
        kwargs.update(simplify=True, dynamic=False, opset=12)
    path = Path(model.export(**kwargs))
    # Sisipkan imgsz ke nama agar export beberapa ukuran tidak saling menimpa.
    # Akhiran "_ncnn_model" wajib dipertahankan: Ultralytics mengenali format
    # NCNN dari akhiran nama folder tersebut.
    stem = Path(args.weights).stem
    if args.format == "ncnn":
        target = path.with_name(f"{stem}_{args.imgsz}_ncnn_model")
    else:
        target = path.with_name(f"{stem}_{args.imgsz}{path.suffix}")
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    path = path.rename(target)
    print(f"\nHasil export: {path}")
    print("Salin file/folder tersebut ke Raspberry Pi, lalu jalankan detect.py --model <path>")


if __name__ == "__main__":
    main()
