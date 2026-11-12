# Unit tests — fondasi paket A

Suite ini memakai DataFrame kecil dan direktori temporer pytest. Tidak diperlukan dataset UNSW-NB15 penuh, `dvc pull`, Zeek, Docker, kredensial, atau layanan MLflow. Model tidak dilatih atau didaftarkan selama pengujian.

## Menjalankan lokal

Dari root repo, gunakan lingkungan Python 3.11:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q --junitxml=reports/pytest.xml
```

Jika `.venv` dibuat dengan uv dan tidak memiliki pip, gunakan:

```bash
uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest -q --junitxml=reports/pytest.xml
```

`requirements-test.txt` mengambil versi dependency dari `requirements.txt` sebagai constraints. Hanya dependency yang dibutuhkan untuk mengimpor dan menguji modul yang dipasang; `mlflow-skinny` menyediakan API yang dipakai test tanpa memasang seluruh layanan MLflow.

## Cakupan

- Parsing override konfigurasi dan resolusi URI tracking.
- Encoding kategori baru, pemisahan distribusi training/evaluasi, dan urutan fitur saat inferensi.
- Pemetaan fitur Zeek dengan durasi/paket nol atau hilang.
- Penggabungan dataset dan kesesuaian metrik data yang disimpan.

Paket B menambahkan unit test gate performa, identitas kandidat, probabilitas, dan registrasi Staging. Setelah perbaikan split validasi, suite memiliki **59 kasus uji**: 52 kasus sebelumnya ditambah 7 kasus pemisahan data. Threshold drift dan migrasi storage belum diperlukan.

## Daftar 18 kasus uji paket A

Suite awal paket A berisi **13 fungsi test yang menghasilkan 18 kasus uji**. Beberapa fungsi memakai `pytest.mark.parametrize`, sehingga perilaku yang sama diperiksa pada beberapa tipe input. Angka `18 passed` pada bukti paket A berarti seluruh kasus uji awal tersebut lulus, bukan 18 model dilatih.

| No. | File | Perilaku yang diperiksa |
|---|---|---|
| 1 | [test_config.py](test_config.py) | Override parameter membaca angka, boolean, null, dan daftar dengan benar tanpa mengubah file konfigurasi asli atau default pada pemanggilan berikutnya. |
| 2 | [test_config.py](test_config.py) | Override tanpa tanda `=` ditolak. |
| 3 | [test_config.py](test_config.py) | URI SQLite relatif tetap menunjuk database di root proyek meskipun folder kerja berubah. |
| 4 | [test_config.py](test_config.py) | URL server MLflow dipertahankan. |
| 5 | [test_config.py](test_config.py) | URI SQLite dengan path absolut dipertahankan. |
| 6 | [test_config.py](test_config.py) | Konfigurasi bertingkat diratakan tanpa kehilangan nilai, termasuk daftar dan boolean. |
| 7 | [test_encoding.py](test_encoding.py) | Kolom kategori bertipe `object` memakai distribusi training; kategori baru menjadi 0 tanpa mengubah input. |
| 8 | [test_encoding.py](test_encoding.py) | Perilaku encoding yang sama untuk kolom bertipe `string`. |
| 9 | [test_encoding.py](test_encoding.py) | Perilaku encoding yang sama untuk kolom bertipe `category`. |
| 10 | [test_encoding.py](test_encoding.py) | Input yang seluruhnya numerik tetap mempertahankan nilai dan kolomnya. |
| 11 | [test_encoding.py](test_encoding.py) | Encoding saat inferensi sesuai hasil encoder training dan urutan fitur benar meskipun urutan kolom input berbeda. |
| 12 | [test_encoding.py](test_encoding.py) | Input inferensi yang kehilangan fitur numerik wajib ditolak. |
| 13 | [test_preprocessing.py](test_preprocessing.py) | Durasi/paket nol atau hilang tidak menghasilkan NaN/infinity pada fitur turunan; nilai hasil perhitungan dan input asli diperiksa. |
| 14 | [test_preprocessing.py](test_preprocessing.py) | Perhitungan waktu akhir, kategori port, hitungan koneksi, dan pengisian label sesuai input. |
| 15 | [test_dataset_io.py](test_dataset_io.py) | Data baru digabung dengan dataset terbaru jika tersedia, meskipun base training juga tersedia. |
| 16 | [test_dataset_io.py](test_dataset_io.py) | Jika dataset terbaru belum tersedia, penggabungan menggunakan base training. |
| 17 | [test_dataset_io.py](test_dataset_io.py) | Jika dataset terbaru dan base training belum tersedia, data baru tetap tersimpan dengan benar. |
| 18 | [test_dataset_io.py](test_dataset_io.py) | Penyimpanan tanpa opsi merge mengganti dataset sebelumnya. |

Kasus 15–17 juga memeriksa bahwa input tidak berubah serta jumlah baris, jumlah kolom, dan distribusi label pada `dataset_metrics.json` sesuai data yang disimpan. Kasus 14 hanya memeriksa pengisian label yang diminta; test tersebut tidak membuktikan label default merupakan ground truth yang benar.

Ringkasan jumlah: konfigurasi **6**, encoding/inferensi **6**, preprocessing **2**, dan penyimpanan dataset **4**.

## Apa yang dijalankan pada paket B?

**Seluruh unit test paket A tetap dijalankan.** Paket B memperluas pengujian dan menambahkan tahapan pipeline sesudah pytest lolos; unit test yang sudah ada tidak diganti dengan evaluasi performa model.

| Lapisan | Yang dijalankan | Data/layanan | Status |
|---|---|---|---|
| Unit test paket A | Pemeriksaan perilaku kode pada 18 kasus di atas. | Fixture kecil, direktori temporer; tanpa layanan eksternal. | Sudah tersedia. |
| Unit test tambahan paket B | Pemeriksaan validator: batas threshold, metrik hilang/NaN/infinity, serta penolakan registrasi jika validasi gagal atau identitas kandidat tidak cocok. | Fixture dan mock client registry. | Sudah tersedia, terutama [test_validation.py](test_validation.py). |
| Pemeriksaan data pipeline | Validasi schema/label dan pemisahan training/evaluasi setelah data diambil dari DVC. | Snapshot data DVC yang dipatok. | Sudah diuji lokal. |
| Evaluasi model | Pengukuran Recall, F1, dan PR-AUC yang disepakati, lalu perbandingan dengan threshold kelayakan Staging. | Kandidat model hasil training dan data evaluasi berlabel. | Sudah diuji lokal. |
| Verifikasi registry | Pembacaan ulang versi model yang diregistrasi untuk memastikan status Staging dan alias yang diharapkan. | MLflow registry dan artifact store. | Sudah diuji pada database lokal terpisah. |

Alur yang ditargetkan:

```text
Push/PR
  → pytest: unit test paket A + tambahan paket B
  → jika lolos: ambil data DVC dan periksa kesiapan data
  → training kandidat
  → evaluasi dan validasi threshold
  → jika lolos, pada push tepercaya ke main: register ke Staging
```

Suite paket B memiliki 59 kasus uji, termasuk 18 kasus awal paket A. Hasil pytest menunjukkan perilaku kode yang diuji berjalan sesuai harapan; kelayakan model ditunjukkan oleh laporan metrik dan keputusan gate evaluasi. Threshold pemicu drift tetap menjadi cakupan paket C/D, terpisah dari threshold kelayakan model pada paket B. Petunjuk setup dan bukti pipeline tersedia di [panduan paket B](../docs/code-as-trigger/04_PAKET_B.md).

### Tujuh kasus tambahan untuk split validasi

File [test_training_split.py](test_training_split.py) memeriksa:

1. Split dapat direproduksi, mencakup seluruh baris, memiliki kedua kelas, dan menjaga semua salinan flow pada sisi yang sama.
2. Validasi kesiapan data berhasil meskipun direktori hidden test tidak tersedia.
3. Evaluator membaca hanya partisi validasi, dengan identitas snapshot yang sama dengan pemeriksaan data.
4. Perubahan seed mengubah identitas evaluasi.
5. Perubahan rasio split mengubah identitas evaluasi.
6. Sumber evaluasi yang tidak dikenal ditolak.
7. Split yang kehilangan salah satu kelas ditolak.

Gate sekarang memakai split `GroupShuffleSplit` dari snapshot DVC `dataset.csv`, dengan porsi 20% kelompok flow dan seed 42. Jumlah baris aktual bisa sedikit berbeda dari 20%. Batch 7 disimpan untuk pengujian akhir dan tidak dibaca oleh gate otomatis.

## GitHub Actions

[Workflow MLOps Automation](../.github/workflows/mlops-automation.yaml) menjalankan suite pada GitHub-hosted runner ketika ada push ke `main`, pull request yang menargetkan `main`, atau pemicu manual dari tab Actions. Kegagalan pytest membuat job gagal; report JUnit `reports/pytest.xml` diunggah sebagai artifact meskipun test gagal.

Workflow baru mulai berjalan di GitHub setelah file-file ini di-commit dan di-push. Hasil lokal bukan bukti run Actions sukses. Runner testing tidak mengakses storage SeaweedFS lokal.

## Verifikasi lokal

Suite awal paket A menghasilkan **18 test lulus**. Setelah implementasi paket B, suite menghasilkan **52 test lulus** pada lingkungan Python 3.11.15 terpisah. Report JUnit lokal paket B tersedia pada `reports/package-b-unit/pytest.xml` dan diabaikan Git. Uji unit tidak melatih model; uji integrasi lokal pada data penuh dijalankan terpisah. Hasil GitHub Actions paket B perlu diperiksa setelah setup runner dan perubahan di-push.
