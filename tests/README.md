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

Pengujian gate performa, training otomatis, dan registrasi Staging ditambahkan pada paket B setelah komponennya tersedia. Threshold drift dan migrasi storage belum diperlukan.

## GitHub Actions

[Workflow MLOps Automation](../.github/workflows/mlops-automation.yaml) menjalankan suite pada GitHub-hosted runner ketika ada push ke `main`, pull request yang menargetkan `main`, atau pemicu manual dari tab Actions. Kegagalan pytest membuat job gagal; report JUnit `reports/pytest.xml` diunggah sebagai artifact meskipun test gagal.

Workflow baru mulai berjalan di GitHub setelah file-file ini di-commit dan di-push. Hasil lokal bukan bukti run Actions sukses. Runner testing tidak mengakses storage SeaweedFS lokal.

## Verifikasi lokal

Suite telah dijalankan pada lingkungan Python 3.11.15 terpisah dengan dependency dari `requirements-test.txt`: **18 test lulus**. Report JUnit dihasilkan pada `reports/pytest.xml` dan diabaikan Git. Hasil GitHub Actions perlu diperiksa setelah perubahan di-push.
