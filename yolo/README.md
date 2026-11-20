# NutriScan – Deteksi menu MBG (YOLO-seg nano, Raspberry Pi 4 CPU)

Model: **YOLO segmentation varian nano** (default `yolo26n-seg`, alternatif `yolo11n-seg`).
Segmentasi dipilih (bukan sekadar bounding box) karena briefing membagi berat load
cell berdasarkan **proporsi luas mask** tiap makanan.

| Langkah | Di mana | Skrip |
|---|---|---|
| 1. Konversi COCO → YOLO-seg + perbaikan rotasi | laptop | `prepare_dataset.py` |
| 1b. Split ulang per sesi foto (anti-bocor) | laptop | `resplit.py` |
| 2. Training | laptop GPU / Colab | `train.py` |
| 3. Export ke NCNN | laptop | `export_model.py` |
| 4. Inferensi + estimasi gram & gizi | Raspberry Pi 4 | `detect.py` |

## 1–3. Di laptop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt

python prepare_dataset.py --src "../menu-mbg-mbkm.v1i.coco/menu-mbg-mbkm.v1i.coco" \
    --out ../datasets/mbg-yolo --preview 20
python resplit.py --dataset ../datasets/mbg-yolo          # -> data_scene.yaml
python train.py --data ../datasets/mbg-yolo/data_scene.yaml --name seg-n-scene
python export_model.py --weights runs/mbg/seg-n-scene/weights/best.pt --imgsz 640   # hasil: best_640_ncnn_model/
```

**Catatan dataset:** ±50% file JPG tersimpan dengan orientasi berbeda dari
anotasinya (tanpa tag EXIF) dan arah putarnya tidak seragam. `prepare_dataset.py`
memilih rotasi per gambar secara otomatis; cek folder `preview/` dan baris teratas
`rotation_log.csv` (margin paling kecil = keputusan paling tidak yakin).

## 4. Di Raspberry Pi 4

Wajib **Raspberry Pi OS 64-bit**.

```bash
sudo apt install python3-picamera2   # bila memakai Pi Camera
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements-raspi.txt
```

Salin `detect.py`, folder `best_640_ncnn_model/`, dan `classes_nutritions.csv`, lalu:

```bash
python detect.py --model best_640_ncnn_model --source piring.jpg --berat 350 \
    --nutrisi classes_nutritions.csv --save --json hasil.json
python detect.py --model best_640_ncnn_model --source camera --berat 350 --nutrisi classes_nutritions.csv
```

`--imgsz` harus sama dengan saat export. Jika terlalu lambat, export ulang dengan
`--imgsz 480` atau `416` (lebih cepat, akurasi sedikit turun) dan bandingkan.

## Struktur dataset & split

Dataset berisi **10 piring (10 menu) × ±150 foto**: tiap piring fisik difoto berulang
di 2–4 latar (meja kayu, kain hitam, batik, taplak renda, kursi kulit, ...). Split
bawaan Roboflow menyebar foto berurutan secara acak, sehingga 297/310 gambar val dan
145/154 gambar test punya "kembaran" (piring+latar+sudut hampir sama) di train.

`resplit.py` membagi ulang per **sesi = piring + latar**: tiap piring dipecah menurut
latarnya, lalu latar-latar itu dibagi ke train/val/test. Gambar test selalu berasal dari
latar yang tidak pernah dilihat model untuk piring tersebut. Hasil: train 834 / val 314 /
test 400 gambar; semua 39 kelas ada di train dan test (4 kelas piring 9 tidak ada di val
karena piring itu hanya punya 2 latar). Audit per gambar: `scene_groups.csv`.

## Hasil training (19 Nov 2026)

`yolo26n-seg`, imgsz 640, laptop RTX 5050.

| Model | Data test | Mask mAP50 | Mask mAP50-95 | Waktu/gambar, CPU laptop 4 core* |
|---|---|---|---|---|
| **Split per sesi** (`runs/mbg/seg-n-scene`) – PyTorch | 400 gambar, latar baru | **0.975** | **0.893** | – |
| Split per sesi – NCNN 640 | idem | 0.975 | 0.892 | 101 ms |
| Split per sesi – NCNN 416 | idem | 0.969 | 0.830 | 48 ms |
| Split Roboflow (`runs/mbg/seg-n`) – PyTorch | 154 gambar, bocor | 0.959 | 0.886 | – |

\* Bukan angka Raspberry Pi; CPU Pi 4 jauh lebih lambat, wajib diukur di perangkat.

Kelas terlemah (split per sesi, mask mAP50-95): Bakso Saus BBQ 0.55, Anggur 0.71,
Kelengkeng 0.72 (dianotasi kadang per butir, kadang per kelompok).

**Arti angka ini:** model terbukti tahan terhadap **latar & sudut kamera baru**. Namun
seluruh test tetap memakai **makanan fisik yang sama** dengan train (hanya ada 1 piring
per menu), sehingga dataset ini *tidak bisa* mengukur akurasi pada piring/porsi/masakan
baru. Untuk itu perlu foto piring MBG lain (hari/dapur berbeda) sebagai test set.

## Keterbatasan (sesuai briefing)

- Estimasi gram = proporsi luas mask × berat total; ini baseline, bukan pengukuran
  massa. Ketebalan/kepadatan makanan tidak diketahui dari satu foto.
- Nilai gizi diambil dari `classes_nutritions.csv` (per 100 g); validasi sumbernya
  tetap tanggung jawab tim.
