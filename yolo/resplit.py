"""
Split ulang dataset per SESI FOTO agar evaluasi tidak bocor.

Masalah split bawaan Roboflow: foto bernomor berurutan (IMG_7700, 7701, ...)
adalah piring yang SAMA yang difoto ulang dari sudut sedikit berbeda, dan
Roboflow menyebarnya acak ke train/val/test. Hasilnya ~95% gambar val/test punya
"kembaran" di train sehingga mAP terlalu optimis.

Struktur dataset ini: tiap piring (menu sama, makanan fisik sama) difoto puluhan
kali di beberapa latar (meja kayu, kain hitam, batik, taplak, ...). Banyak kelas
hanya muncul di 1-2 piring, jadi menahan piring utuh untuk test tidak mungkin.
Kompromi yang dipakai di sini: unit split = SESI = (piring, latar). Gambar test
selalu berasal dari kombinasi piring+latar yang tidak pernah dilihat saat
training; tidak ada lagi foto kembar berurutan yang menyeberang split.

Langkah:
1. Urutkan semua gambar menurut nomor IMG.
2. Batas PIRING: komposisi menu berubah (Jaccard < 0.6 terhadap menu dominan).
3. Batas LATAR di dalam piring: histogram warna HSV rata-rata 3 foto sebelum
   vs 3 foto sesudah berubah tajam (puncak lokal > --scene-thresh).
4. Segmen di piring yang sama dengan latar mirip (jarak < --merge-thresh)
   digabung jadi satu sesi, karena latar yang sama bisa terpotong beberapa kali
   oleh perubahan sudut kamera.
   Bila sebuah piring masih punya < 3 sesi, ambang diturunkan bertahap (sampai
   MIN_THRESH). Piring yang memang hanya punya 2 latar dibagi ke train + test.
5. Sesi TIAP PIRING dibagi ke train/val/test mendekati rasio target. Karena
   satu piring = satu menu, semua makanannya hadir di ketiga split.

Gambar tidak disalin; skrip menulis train.txt / val.txt / test.txt dan
data_scene.yaml di folder dataset, plus scene_groups.csv untuk audit.

Contoh:
    python resplit.py --dataset ../datasets/mbg-yolo
    python train.py --data ../datasets/mbg-yolo/data_scene.yaml --name seg-n-scene
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

WINDOW = 3
MIN_SEG = 8  # segmen latar minimal (foto)
MIN_THRESH = 0.3  # batas bawah penurunan ambang; di bawah ini latar yang sama ikut terpecah
SPLITS = ("train", "val", "test")


def hsv_hist(path):
    im = Image.open(path).convert("HSV")
    im.thumbnail((96, 96))
    a = np.asarray(im).reshape(-1, 3)
    h, _ = np.histogramdd(a, bins=(12, 4, 4), range=((0, 256),) * 3)
    return (h / h.sum()).ravel()


def l1(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def find_plates(classes, min_size=3):
    """Index awal tiap piring (urutan sudah per nomor IMG).

    Potongan < min_size gambar biasanya foto yang anotasinya kurang lengkap
    (mis. hanya 2 dari 5 makanan dilabeli), bukan piring baru: digabung ke
    piring sebelumnya, lalu piring sesudahnya ikut digabung bila menunya sama
    (Jaccard >= 0.8: identik atau hanya kurang satu item).
    """
    starts, cur = [0], [classes[0]]
    for i in range(1, len(classes)):
        recent = cur[-5:]
        cnt = Counter(c for s in recent for c in s)
        dominant = frozenset(c for c, n in cnt.items() if n >= max(1, len(recent) // 2))
        if jaccard(classes[i], dominant) < 0.6:
            starts.append(i)
            cur = []
        cur.append(classes[i])

    bounds = starts + [len(classes)]
    plates = [list(range(bounds[j], bounds[j + 1])) for j in range(len(starts))]

    def menu(idx):
        cnt = Counter(c for i in idx for c in classes[i])
        return frozenset(c for c, n in cnt.items() if n >= len(idx) / 2)

    merged = [plates[0]]
    for pl in plates[1:]:
        if len(pl) < min_size or jaccard(menu(pl), menu(merged[-1])) >= 0.8:
            merged[-1].extend(pl)
        else:
            merged.append(pl)
    # potongan kecil di paling awal
    if len(merged) > 1 and len(merged[0]) < min_size:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return [pl[0] for pl in merged]


def scene_scores(feats):
    """Skor perubahan latar di tiap posisi: jarak histogram rata-rata
    WINDOW foto sebelum vs WINDOW foto sesudah."""
    n = len(feats)
    score = np.zeros(n)
    for i in range(1, n):
        score[i] = l1(feats[max(0, i - WINDOW):i].mean(0), feats[i:i + WINDOW].mean(0))
    return score


def plate_sessions(idx, feats, score, scene_thresh, merge_thresh, min_sessions):
    """Bagi satu piring (index berurutan) menjadi sesi latar.

    Mulai dari ambang default; bila sesi < min_sessions, ambang gabung lalu
    ambang deteksi diturunkan bertahap (mis. taplak renda di atas meja kayu
    histogramnya mirip dengan meja kayu saja).
    """
    lo, hi = idx[0], idx[-1] + 1
    result = [list(idx)]
    for st in np.arange(scene_thresh, MIN_THRESH - 1e-9, -0.05):
        local = score[lo:hi]
        cand = [i for i in range(1, hi - lo)
                if local[i] > st and local[i] == local[max(1, i - WINDOW):i + WINDOW + 1].max()]
        # segmen minimal MIN_SEG foto: ambil batas terkuat lebih dulu
        kept = []
        for i in sorted(cand, key=lambda i: -local[i]):
            if min(i, hi - lo - i) >= MIN_SEG and all(abs(i - k) >= MIN_SEG for k in kept):
                kept.append(i)
        breaks = [lo + i for i in sorted(kept)]
        bounds = [lo] + breaks + [hi]
        segs = [list(range(bounds[j], bounds[j + 1])) for j in range(len(bounds) - 1)]
        means = [feats[sg].mean(0) for sg in segs]
        for mt in np.arange(merge_thresh, MIN_THRESH - 1e-9, -0.05):
            parent = list(range(len(segs)))

            def root(x):
                while parent[x] != x:
                    x = parent[x]
                return x

            for a in range(len(segs)):
                for b in range(a + 1, len(segs)):
                    if l1(means[a], means[b]) < mt:
                        parent[root(a)] = root(b)
            groups = defaultdict(list)
            for j, sg in enumerate(segs):
                groups[root(j)].extend(sg)
            result = sorted(groups.values(), key=lambda g: g[0])
            if len(result) >= min_sessions:
                return result
    return result


def build_sessions(items, feats, scene_thresh, merge_thresh, min_sessions=3):
    """items terurut -> (list sesi, {index: nomor piring}, list piring)."""
    n = len(items)
    starts = find_plates([it["classes"] for it in items])
    bounds = starts + [n]
    plates = [list(range(bounds[j], bounds[j + 1])) for j in range(len(starts))]
    score = scene_scores(feats)
    sessions, plate_of_sess = [], []
    for p, idx in enumerate(plates):
        for sess in plate_sessions(idx, feats, score, scene_thresh, merge_thresh, min_sessions):
            sessions.append(sess)
            plate_of_sess.append(p)
    plate_of = {i: plate_of_sess[j] for j, sess in enumerate(sessions) for i in sess}
    return sessions, plate_of, plate_of_sess


def assign_splits(sessions, plate_of_sess, items, nc, ratios):
    """Bagi sesi tiap piring ke train/val/test.

    Per piring dicoba semua kemungkinan penempatan sesinya; dipilih yang ketiga
    split terisi (bila sesinya >= 3), train paling besar, dan proporsi jumlah
    foto paling dekat rasio target. Karena tiap piring = satu menu, kelima
    makanannya otomatis muncul di train, val, dan test.
    """
    from itertools import product
    r = np.array(ratios) / sum(ratios)
    assign = {}
    by_plate = defaultdict(list)
    for j, p in enumerate(plate_of_sess):
        by_plate[p].append(j)
    for p, js in by_plate.items():
        sizes = np.array([len(sessions[j]) for j in js], float)
        need_all = len(js) >= 3
        best, best_dev = None, 1e18
        for combo in product(range(3), repeat=len(js)):
            combo = np.array(combo)
            got = np.array([sizes[combo == k].sum() for k in range(3)])
            if need_all and (got == 0).any():
                continue
            if got[0] < got[1:].max():
                continue  # train harus bagian terbesar
            dev = np.abs(got / sizes.sum() - r).sum()
            dev += 0.01 * (got[2] == 0)  # piring 2 sesi: utamakan test daripada val
            if dev < best_dev:
                best, best_dev = combo, dev
        for j, k in zip(js, best):
            assign[j] = SPLITS[k]

    got_cls = {s: np.zeros(nc) for s in SPLITS}
    for j, sess in enumerate(sessions):
        for i in sess:
            for k in items[i]["classes"]:
                got_cls[assign[j]][k] += 1
    return assign, got_cls


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="../datasets/mbg-yolo",
                    help="output prepare_dataset.py (berisi images/, labels/, data.yaml)")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--scene-thresh", type=float, default=0.4)
    ap.add_argument("--merge-thresh", type=float, default=0.5)
    ap.add_argument("--min-sessions", type=int, default=3,
                    help="minimal jumlah sesi latar per piring (agar bisa masuk 3 split)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    yaml_text = (root / "data.yaml").read_text()
    names_block = yaml_text[yaml_text.index("nc:"):]
    names = [line.split(": ", 1)[1] for line in names_block.splitlines()[2:] if line.strip()]
    nc = len(names)

    items = []
    for img in sorted((root / "images").glob("*/*.jpg")):
        lbl = root / "labels" / img.parent.name / f"{img.stem}.txt"
        m = re.match(r"IMG_(\d+)", img.name)
        classes = frozenset(int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip())
        items.append({"path": img, "num": int(m.group(1)) if m else 10**9,
                      "orig": img.parent.name, "classes": classes})
    items.sort(key=lambda it: (it["num"], it["path"].name))
    print(f"{len(items)} gambar, {nc} kelas")

    with Pool(args.workers) as pool:
        feats = np.array(pool.map(hsv_hist, [it["path"] for it in items], chunksize=16))

    sessions, plate_of, plate_of_sess = build_sessions(
        items, feats, args.scene_thresh, args.merge_thresh, args.min_sessions)
    n_plates = len(set(plate_of.values()))
    sizes = sorted(len(s) for s in sessions)
    print(f"{n_plates} piring, {len(sessions)} sesi (piring+latar); "
          f"ukuran sesi median {int(np.median(sizes))}, maks {sizes[-1]}")

    assign, got_cls = assign_splits(sessions, plate_of_sess, items, nc, args.ratios)

    lists = defaultdict(list)
    rows = []
    for j, s in enumerate(sessions):
        for i in s:
            lists[assign[j]].append(str(items[i]["path"]))
            rows.append({"image": items[i]["path"].name, "img_num": items[i]["num"],
                         "plate": plate_of[i], "session": j, "split": assign[j],
                         "roboflow_split": items[i]["orig"]})
    for s in SPLITS:
        (root / f"{s}.txt").write_text("\n".join(sorted(lists[s])) + "\n")
    with open(root / "scene_groups.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: (r["img_num"], r["image"])))

    out_yaml = [f"path: {root}", "train: train.txt", "val: val.txt", "test: test.txt"]
    out_yaml += names_block.splitlines()
    (root / "data_scene.yaml").write_text("\n".join(out_yaml) + "\n")

    # --- laporan ---
    print("\nJumlah gambar:", {s: len(lists[s]) for s in SPLITS},
          "| sesi:", dict(Counter(assign.values())))
    # kebocoran: tetangga nomor +-3 di split lain DAN sesi yang sama (harus 0)
    by_num = {(r["img_num"]): r for r in rows}
    near_other = sum(1 for r in rows if r["split"] != "train" and any(
        (n := by_num.get(r["img_num"] + d)) and n["split"] == "train"
        for d in (-3, -2, -1, 1, 2, 3)))
    print(f"Gambar val/test bertetangga (nomor ±3) dengan train: {near_other} "
          f"(hanya di batas sesi; sebelumnya ~440)")
    weak = [(names[k], [int(got_cls[s][k]) for s in SPLITS]) for k in range(nc)
            if min(got_cls["val"][k], got_cls["test"][k]) == 0]
    if weak:
        print("Kelas yang tidak ada di val atau test (train, val, test):")
        for n, c in weak:
            print(f"  {n:<22} {c}")
    print(f"\nSelesai -> {root / 'data_scene.yaml'}  (audit: scene_groups.csv)")


if __name__ == "__main__":
    main()
