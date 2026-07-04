- Install Python 3.10

- Buka terminal di folder project

- Buat virtual environment dengan Python 3.10:
  
py -3.10 -m venv .venv

- Aktifkan environment:
  
.venv\Scripts\activate

- Install dependency:
  
pip install -r requirement.txt

- Jalankan aplikasi dengan Streamlit, bukan langsung tombol Run Python:
  
streamlit run main.py

- Kalau streamlit tidak dikenali:
  
python -m streamlit run main.py
