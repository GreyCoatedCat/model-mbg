"""
Konversi dataset Roboflow COCO (menu-mbg-mbkm) -> format YOLO segmentation.

Yang dilakukan:
1. Membaca _annotations.coco.json tiap split (train/valid/test).
2. Membuang kategori dummy "objects" (id 0) dan memetakan id 1..39 -> 0..38.
3. Memperbaiki orientasi gambar. Sekitar separuh file JPG tersimpan dalam
   orientasi berbeda dari anotasinya (mis. file 4032x3024, anotasi 3024x4032)
   tanpa tag EXIF, dan arah putarnya tidak seragam (ada yang perlu diputar
   searah jarum jam, ada yang berlawanan, ada juga yang terbalik 180 derajat
   meski ukurannya sama). Skrip ini mencoba setiap rotasi yang mungkin lalu
   memilih rotasi yang isi polygon-nya paling homogen dan kontras dengan
   sekitarnya (lihat region_score).
4. Mengecilkan gambar (default sisi terpanjang 1280 px) supaya training jauh
   lebih cepat dan hemat disk; koordinat label dinormalisasi 0..1 sehingga
   tidak terpengaruh resize.
5. Menulis data.yaml untuk Ultralytics + rotation_log.csv untuk audit.

Contoh:
    python prepare_dataset.py \
        --src "../menu-mbg-mbkm.v1i.coco/menu-mbg-mbkm.v1i.coco" \
        --out ../datasets/mbg-yolo --preview 30
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SPLITS = {"train": "train", "valid": "val", "test": "test"}
SCORE_SIZE = 512  # resolusi kecil untuk menilai rotasi
WEAK_MARGIN = 1.5  # di bawah ini keputusan rotasi dianggap kurang yakin


def region_score(img: Image.Image, polys, ann_w, ann_h) -> float:
    """Seberapa cocok polygon dengan isi gambar pada orientasi ini.

    Untuk tiap polygon: beda warna rata-rata antara bagian DALAM polygon dan
    cincin tipis di LUAR polygon, dibagi keragaman warnanya (mirip kriteria
    Fisher). Orientasi yang benar -> isi polygon homogen (satu jenis makanan)
    dan kontras dengan sekitarnya. Lebih tahan terhadap latar bermotif
    (karpet/taplak) dan bentuk ompreng yang simetris dibanding skor tepi.
    """
    w, h = img.size
    a = np.asarray(img.convert("RGB").filter(ImageFilter.GaussianBlur(1)), np.float32)
    total = weight = 0.0
    for p in polys:
        xy = np.asarray(p, np.float32).reshape(-1, 2) * [w / ann_w, h / ann_h]
        if len(xy) < 3:
            continue
        m = Image.new("L", (w, h))
        ImageDraw.Draw(m).polygon([tuple(q) for q in xy], fill=255)
        inside = np.asarray(m) > 0
        dilated = np.asarray(m.filter(ImageFilter.MaxFilter(7))) > 0
        eroded = np.asarray(m.filter(ImageFilter.MinFilter(7))) > 0
        ring = dilated & ~inside
        inner = eroded if eroded.sum() > 20 else inside
        if inner.sum() < 10 or ring.sum() < 10:
            continue
        ci, cr = a[inner], a[ring]
        dist = np.linalg.norm(ci.mean(0) - cr.mean(0))
        score = dist / (ci.std(0).mean() + cr.std(0).mean() + 1e-3)
        wt = np.sqrt(inner.sum())  # objek besar sedikit lebih berpengaruh
        total += score * wt
        weight += wt
    return total / weight if weight else 0.0


def process_image(job):
    src_path, dst_img, dst_lbl, ann_w, ann_h, polys_by_cls, max_size = job
    img = Image.open(src_path)
    img.draft("RGB", (max_size, max_size))  # decode JPEG lebih cepat
    img = img.convert("RGB")

    all_polys = [p for _, p in polys_by_cls]
    w, h = img.size
    transposed = (w > h) != (ann_w > ann_h) and w != h
    candidates = [-90, 90] if transposed else [0, 180]

    small = img.copy()
    small.thumbnail((SCORE_SIZE, SCORE_SIZE))
    scores = {}
    if all_polys:
        for a in candidates:
            rot = small.rotate(a, expand=True) if a else small
            scores[a] = region_score(rot, all_polys, ann_w, ann_h)
    angle = max(scores, key=scores.get) if scores else candidates[0]

    if angle:
        img = img.rotate(angle, expand=True)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img.save(dst_img, quality=92)

    lines = []
    for cls, poly in polys_by_cls:
        xy = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if len(xy) < 3:
            continue
        xy[:, 0] /= ann_w
        xy[:, 1] /= ann_h
        xy = np.clip(xy, 0.0, 1.0)
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xy.ravel()))
    Path(dst_lbl).write_text("\n".join(lines) + ("\n" if lines else ""))

    s = sorted(scores.values(), reverse=True)
    margin = s[0] / s[1] if len(s) == 2 and s[1] > 0 else float("inf")
    return {
        "image": Path(dst_img).name,
        "transposed": transposed,
        "angle": angle,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "margin": round(margin, 3),
    }


def draw_preview(img_path, lbl_path, names, out_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    for line in Path(lbl_path).read_text().splitlines():
        v = line.split()
        xy = np.asarray(v[1:], dtype=float).reshape(-1, 2) * [w, h]
        d.polygon([tuple(p) for p in xy], outline=(255, 0, 0), width=4)
        d.text(tuple(xy[0]), names[int(v[0])], fill=(255, 255, 0))
    img.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder export Roboflow COCO")
    ap.add_argument("--out", default="../datasets/mbg-yolo")
    ap.add_argument("--max-size", type=int, default=1280,
                    help="sisi terpanjang gambar keluaran (px)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preview", type=int, default=0,
                    help="simpan N gambar overlay per split untuk dicek manual")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out).resolve()
    names = None
    log_rows = []

    for coco_split, yolo_split in SPLITS.items():
        ann_file = src / coco_split / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"[skip] {ann_file} tidak ada")
            continue
        coco = json.loads(ann_file.read_text())

        cats = sorted((c for c in coco["categories"] if c["supercategory"] != "none"),
                      key=lambda c: c["id"])
        cat_map = {c["id"]: i for i, c in enumerate(cats)}
        split_names = [c["name"] for c in cats]
        if names is None:
            names = split_names
        elif names != split_names:
            raise SystemExit(f"Daftar kelas split {coco_split} berbeda dengan train!")

        anns = {}
        for a in coco["annotations"]:
            if a["category_id"] not in cat_map or not isinstance(a.get("segmentation"), list):
                continue
            for poly in a["segmentation"]:
                anns.setdefault(a["image_id"], []).append((cat_map[a["category_id"]], poly))

        img_dir = out / "images" / yolo_split
        lbl_dir = out / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        for im in coco["images"]:
            stem = Path(im["file_name"]).stem
            jobs.append((
                str(src / coco_split / im["file_name"]),
                str(img_dir / f"{stem}.jpg"),
                str(lbl_dir / f"{stem}.txt"),
                im["width"], im["height"],
                anns.get(im["id"], []),
                args.max_size,
            ))

        with Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(process_image, jobs, chunksize=4), 1):
                results.append(r)
                if i % 100 == 0 or i == len(jobs):
                    print(f"  {coco_split}: {i}/{len(jobs)}", end="\r")
        print()

        rot = sum(1 for r in results if r["angle"])
        weak = [r for r in results if r["margin"] < WEAK_MARGIN]
        print(f"[{coco_split}] {len(results)} gambar, {rot} diputar, "
              f"{len(weak)} keputusan rotasi kurang yakin (cek rotation_log.csv)")
        for r in results:
            log_rows.append({"split": yolo_split, **r})

        if args.preview:
            prev_dir = out / "preview" / yolo_split
            prev_dir.mkdir(parents=True, exist_ok=True)
            # utamakan gambar yang diputar / kurang yakin
            pick = sorted(results, key=lambda r: (not r["angle"], r["margin"]))[:args.preview]
            for r in pick:
                stem = Path(r["image"]).stem
                draw_preview(img_dir / r["image"], lbl_dir / f"{stem}.txt", names,
                             prev_dir / f"{r['angle']:+d}_{stem}.jpg")

    with open(out / "rotation_log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["split", "image", "transposed", "angle",
                                           "scores", "margin"])
        wr.writeheader()
        wr.writerows(sorted(log_rows, key=lambda r: r["margin"]))

    yaml_lines = [f"path: {out}", "train: images/train", "val: images/val",
                  "test: images/test", f"nc: {len(names)}", "names:"]
    yaml_lines += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"Selesai -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
