Berikut ini README lengkap dalam format markdown yang sudah saya tambahkan semua sesuai permintaanmu (logo/logo badge, link data CSV, dan info akun Metabase):

```markdown
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)

## Business Understanding

Jaya Jaya Institut adalah sebuah institusi pendidikan berbasis teknologi (edutech) yang mengalami permasalahan dalam mempertahankan siswanya. Terjadi peningkatan signifikan dalam jumlah siswa yang mengalami dropout dari tahun ke tahun. Untuk itu, dibutuhkan sebuah sistem yang mampu memprediksi siswa mana yang berpotensi dropout agar pihak institusi dapat mengambil tindakan preventif lebih awal.

### Permasalahan Bisnis

- Tingginya angka dropout siswa dari semester ke semester.
- Belum adanya sistem yang bisa membantu pihak akademik dalam memantau risiko siswa yang berpotensi keluar.
- Kurangnya wawasan berbasis data yang bisa digunakan manajemen untuk melakukan intervensi tepat sasaran.

### Cakupan Proyek

- Melakukan analisis data historis performa siswa.
- Membangun model machine learning untuk memprediksi potensi siswa dropout.
- Membuat prototype sistem prediksi berbasis web menggunakan Streamlit.
- Membangun dashboard visualisasi performa siswa dan prediksi dropout menggunakan Metabase.
- Menyimpan hasil prediksi ke dalam database PostgreSQL.

### Persiapan

Sumber data:  
[students_performance/data.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)  

Setup environment:
```bash
pip install -r requirements.txt
```

---

## Business Dashboard

Dashboard ini dibangun menggunakan Metabase yang terhubung ke database PostgreSQL. Dashboard ini menampilkan informasi visual terkait:

- Jumlah siswa dropout vs non-dropout per semester
- Distribusi performa akademik siswa
- Analisis berdasarkan gender, tingkat pendidikan orang tua, dan faktor lainnya
- Daftar siswa dengan prediksi risiko dropout

### Akun Metabase
- Email: `root@mail.com`
- Password: `root123`

---

## Menjalankan Sistem Machine Learning

Sistem prototype ini dibangun menggunakan Streamlit dan dapat digunakan oleh pihak akademik untuk memprediksi status dropout siswa berdasarkan data input.

Cara menjalankan:
```bash
streamlit run streamlit_app.py
```

🌐 Link Akses Prototype: [https://zidannnapp-dropout.streamlit.app](https://zidannnapp-dropout.streamlit.app)

---

## Conclusion

Proyek ini berhasil membangun sistem prediksi dropout dengan akurasi mencapai **77%**. Model XGBoost terbukti memberikan performa terbaik berdasarkan metrik akurasi dan F1-score. Dashboard Metabase berhasil menyediakan insight visual untuk mendukung keputusan manajerial secara data-driven.

### Rekomendasi Action Items

- Menggunakan hasil prediksi dropout untuk melakukan intervensi akademik lebih awal.
- Meningkatkan komunikasi dan bimbingan terhadap siswa dengan risiko tinggi.
- Melakukan monitoring berkala terhadap faktor-faktor yang paling berpengaruh terhadap dropout.
- Melanjutkan pengembangan fitur seperti integrasi dengan sistem akademik internal dan notifikasi otomatis.
```