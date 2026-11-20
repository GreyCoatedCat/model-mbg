"""
Deteksi + segmentasi menu MBG untuk Raspberry Pi 4 (CPU saja, tanpa akselerator).

Alur (sesuai briefing NutriScan):
    foto piring -> YOLO-seg (mask per makanan) -> proporsi luas mask
    -> berat total load cell dibagi per komponen -> estimasi gizi

Contoh di Raspi:
    # satu foto
    python detect.py --model best_640_ncnn_model --source piring.jpg --berat 350 --save
    # satu folder
    python detect.py --model best_640_ncnn_model --source foto/ --json hasil.json
    # ambil foto dari kamera (Pi Camera via picamera2, atau webcam USB)
    python detect.py --model best_640_ncnn_model --source camera --berat 350 --save

Bisa juga di-import oleh modul integrasi:
    from detect import load_model, load_nutrition, analyze
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

NUTRISI_KOLOM = {  # kolom CSV -> nama keluaran (nilai per 100 g)
    "energi": "energi_kcal",
    "protein": "protein_g",
    "lemak": "lemak_g",
    "karbohidrat": "karbohidrat_g",
}
DEFAULT_CSV = (Path(__file__).resolve().parent.parent / "menu-mbg-mbkm.v1i.coco"
               / "menu-mbg-mbkm.v1i.coco" / "docs" / "classes_nutritions.csv")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(path, imgsz):
    model = YOLO(str(path), task="segment")
    # warm-up: inferensi pertama di Raspi selalu jauh lebih lambat
    model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)
    return model


def load_nutrition(path):
    """Baca classes_nutritions.csv -> {nama_kelas: {energi_kcal: .., ...}}.

    CSV sumber berisi spasi/tab dan sel kosong yang tidak rapi; kolom gizi makro
    berada di awal baris sehingga tetap terbaca benar meskipun kolom setelahnya
    bergeser.
    """
    if not path or not Path(path).is_file():
        return {}
    table = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip().lower() for h in next(reader)]
        for row in reader:
            row = [c.strip() for c in row]
            if len(row) < len(NUTRISI_KOLOM) + 2:
                continue
            values = {}
            for col, out in NUTRISI_KOLOM.items():
                try:
                    values[out] = float(row[header.index(col)].replace(",", "."))
                except (ValueError, IndexError):
                    values[out] = None
            table[row[header.index("class_name")]] = values
    return table


def capture_camera(width=2304, height=1296):
    """Ambil satu frame (BGR). Coba Pi Camera (picamera2), lalu webcam USB."""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}))  # = urutan BGR
        cam.start()
        time.sleep(1.5)  # beri waktu auto-exposure / white balance
        frame = cam.capture_array("main")
        cam.close()
        return frame
    except ImportError:
        pass
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):  # buang beberapa frame awal (exposure belum stabil)
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Kamera tidak bisa dibaca (picamera2 / /dev/video0)")
    return frame


def analyze(model, image, imgsz=640, conf=0.35, berat_total=None, nutrisi=None):
    """Jalankan model pada satu gambar (path atau array BGR) -> dict hasil."""
    t0 = time.perf_counter()
    res = model.predict(image, imgsz=imgsz, conf=conf, retina_masks=False,
                        verbose=False)[0]
    wall_ms = (time.perf_counter() - t0) * 1000
    names = res.names

    detections = []
    komponen = {}
    if res.boxes is not None and len(res.boxes):
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        for c, p, b in zip(cls, confs, boxes):
            detections.append({"kelas": names[c], "confidence": round(float(p), 3),
                               "bbox_xyxy": [round(float(v), 1) for v in b]})

        if res.masks is not None:
            # Peta label: tiap piksel milik satu kelas saja. Instance dengan
            # confidence tertinggi menang bila mask saling tumpang-tindih,
            # supaya area tidak dihitung dua kali.
            masks = res.masks.data.cpu().numpy() > 0.5
            label_map = np.full(masks.shape[1:], -1, np.int16)
            for i in np.argsort(confs):
                label_map[masks[i]] = cls[i]
            ids, counts = np.unique(label_map[label_map >= 0], return_counts=True)
            total_px = counts.sum()
            for c, n in zip(ids, counts):
                komponen[names[int(c)]] = {"luas_px": int(n),
                                           "proporsi": float(n / total_px)}

    total_gizi = {}
    for nama, k in komponen.items():
        if berat_total is not None:
            k["estimasi_gram"] = round(k["proporsi"] * berat_total, 1)
        k["proporsi"] = round(k["proporsi"], 4)
        if berat_total is not None:
            if nutrisi and nama in nutrisi:
                k["gizi"] = {}
                for key, per100 in nutrisi[nama].items():
                    if per100 is None:
                        continue
                    v = k["estimasi_gram"] / 100 * per100
                    k["gizi"][key] = round(v, 2)
                    total_gizi[key] = round(total_gizi.get(key, 0) + v, 2)

    return {
        "jumlah_deteksi": len(detections),
        "deteksi": detections,
        "komponen": komponen,
        "berat_total_g": berat_total,
        "total_gizi": total_gizi or None,
        "waktu_ms": {**{k: round(v, 1) for k, v in res.speed.items()},
                     "total": round(wall_ms, 1)},
        "_result": res,  # objek Ultralytics, dibuang sebelum disimpan ke JSON
    }


def print_report(name, out):
    print(f"\n=== {name} ===")
    w = out["waktu_ms"]
    print(f"Waktu: {w['total']:.0f} ms (pre {w.get('preprocess', 0):.0f} / "
          f"infer {w.get('inference', 0):.0f} / post {w.get('postprocess', 0):.0f})")
    if not out["komponen"]:
        print("Tidak ada makanan terdeteksi.")
        return
    for d in out["deteksi"]:
        print(f"  - {d['kelas']:<20} conf {d['confidence']:.2f}")
    print("Komponen (proporsi luas mask):")
    for nama, k in sorted(out["komponen"].items(), key=lambda x: -x[1]["proporsi"]):
        line = f"  {nama:<20} {k['proporsi'] * 100:5.1f}%"
        if "estimasi_gram" in k:
            line += f"  ~{k['estimasi_gram']:.0f} g"
        if k.get("gizi", {}).get("energi_kcal") is not None:
            line += f"  ~{k['gizi']['energi_kcal']:.0f} kkal"
        print(line)
    if out["total_gizi"]:
        print("Total gizi (estimasi): " + ", ".join(
            f"{k}={v}" for k, v in out["total_gizi"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="folder *_ncnn_model, file .onnx, atau .pt")
    ap.add_argument("--source", required=True, help="file gambar, folder, atau 'camera'")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="harus sama dengan imgsz saat export NCNN/ONNX")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--berat", type=float, default=None,
                    help="berat bersih total makanan dari load cell (gram)")
    ap.add_argument("--nutrisi", default=str(DEFAULT_CSV),
                    help="path classes_nutritions.csv")
    ap.add_argument("--save", action="store_true", help="simpan gambar hasil anotasi")
    ap.add_argument("--out-dir", default="hasil_deteksi")
    ap.add_argument("--json", default=None, help="simpan hasil ke file JSON")
    args = ap.parse_args()

    nutrisi = load_nutrition(args.nutrisi)
    if args.berat is not None and not nutrisi:
        print(f"[peringatan] tabel gizi tidak ditemukan: {args.nutrisi}")

    t = time.perf_counter()
    model = load_model(args.model, args.imgsz)
    print(f"Model dimuat dalam {time.perf_counter() - t:.1f} s")

    if args.source == "camera":
        sources = [("camera_" + time.strftime("%Y%m%d_%H%M%S"), capture_camera())]
    else:
        p = Path(args.source)
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) \
            if p.is_dir() else [p]
        sources = [(f.stem, str(f)) for f in files]

    out_dir = Path(args.out_dir)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
    semua = []
    for name, img in sources:
        out = analyze(model, img, args.imgsz, args.conf, args.berat, nutrisi)
        res = out.pop("_result")
        print_report(name, out)
        if args.save:
            cv2.imwrite(str(out_dir / f"{name}_hasil.jpg"), res.plot())
            if args.source == "camera":
                cv2.imwrite(str(out_dir / f"{name}.jpg"), img)
        semua.append({"sumber": name, **out})

    if len(semua) > 1:
        t_inf = [s["waktu_ms"]["total"] for s in semua]
        print(f"\nRata-rata waktu per gambar: {np.mean(t_inf):.0f} ms "
              f"({len(t_inf)} gambar)")
    if args.json:
        Path(args.json).write_text(json.dumps(semua if len(semua) > 1 else semua[0],
                                              indent=2, ensure_ascii=False))
        print(f"JSON disimpan: {args.json}")


if __name__ == "__main__":
    main()
