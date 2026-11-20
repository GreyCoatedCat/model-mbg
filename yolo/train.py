"""
Training YOLO segmentation (varian nano) untuk menu MBG. Jalankan di laptop/PC
yang punya GPU (atau Google Colab), BUKAN di Raspberry Pi.

Kenapa segmentasi (bukan hanya bounding box)? Briefing NutriScan membagi berat
total dari load cell berdasarkan proporsi LUAS MASK tiap makanan, jadi model
harus menghasilkan mask. Varian "-seg" ukuran "n" (nano) adalah yang paling
ringan dan masih realistis untuk CPU Raspberry Pi 4.

Contoh:
    python train.py --data ../datasets/mbg-yolo/data.yaml
    python train.py --data ../datasets/mbg-yolo/data.yaml --model yolo11n-seg.pt --imgsz 512
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../datasets/mbg-yolo/data.yaml")
    ap.add_argument("--model", default="yolo26n-seg.pt",
                    help="bobot pretrained COCO; alternatif: yolo11n-seg.pt, yolov8n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="samakan dengan ukuran yang akan dipakai di Raspi")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30, help="early stopping")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' untuk GPU pertama, 'cpu' untuk CPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/mbg")
    ap.add_argument("--name", default="seg-n")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="pakai sebagian data train (mis. 0.1) untuk uji cepat pipeline")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(Path(args.project).resolve()),  # path absolut: cegah runs/segment/ ganda
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        cache="ram",
        # Foto diambil dari atas (top-down), jadi flip vertikal & rotasi masih
        # realistis. Augmentasi warna dibiarkan default agar warna makanan
        # (mis. nasi putih vs nasi daun jeruk) tidak terlalu berubah.
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        plots=True,
    )

    # Evaluasi akhir pada split test (data yang belum pernah dilihat model)
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    metrics = best.val(data=args.data, split="test", imgsz=args.imgsz,
                       device=args.device, project=str(Path(args.project).resolve()),
                       name=f"{args.name}-test", exist_ok=True)
    print("\n=== Hasil split TEST ===")
    print(f"Box  mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"Mask mAP50: {metrics.seg.map50:.3f}   mAP50-95: {metrics.seg.map:.3f}")
    print(f"Bobot terbaik: {model.trainer.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
