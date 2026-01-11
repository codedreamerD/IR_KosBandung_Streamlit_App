TUTORIAL DEPLOY / RUN STREAMLIT (ANTI BUG / ANTI HUMAN ERROR)

0) Struktur folder WAJIB:
IR_EAS_IFB307_KosBandung_Streamlit/
  app.py
  requirements.txt
  data/
    Dataset_ReviewKos_v1.csv

1) Pastikan dataset benar:
- Tepat 30 baris dokumen (D1..D30)
- Kolom minimal: doc_id, review_text
- doc_id unik, review_text tidak kosong

2) Install dependency
A) (Disarankan) bikin virtual environment:
   python -m venv .venv
   .venv\Scripts\activate     (Windows)
   source .venv/bin/activate   (Mac/Linux)

B) Install library:
   pip install -r requirements.txt

3) Jalankan Streamlit
   streamlit run app.py

4) Jika error:
- FileNotFoundError dataset: cek path data/Dataset_ReviewKos_v1.csv
- ValueError jumlah dokumen: pastikan 30
- doc_id duplikat: hapus duplikat di CSV
- review_text kosong: isi / hapus baris kosong

5) Demo yang disarankan (untuk screenshot laporan):
- murah dekat kampus
- kamar bersih aman
- wifi kencang fasilitas lengkap
Ambil screenshot hasil Top-5 + summary dan tampilan inverted index (cek term).
