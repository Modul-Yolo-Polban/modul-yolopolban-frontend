# modul-yolopolban-frontend
Front-end services for face and body recognition using YOLO V8 and Streamlit.

Tested in Anaconda with env Python 3.9.17 version

- git clone this repository
- Cd ke file modul-yolopolban-frontend,
- pip install -r requirements.txt 
- conda install -c conda-forge lap (cannot install lap lib from pip, error occured)
- run service backend https://github.com/Modul-Yolo-Polban/modul-yolopolban-backend
- streamlit run main.py

Akses Lokal kalau run di server.
- di tab baru ssh -N -f -L 8101:192.168.34.201:8501 bimad4@103.209.131.66 -p 8022
- akses http://localhost:8101

## Fitur
- Model Face Recognition, Body Recognition, Sampah Recognition
- Detect by image, videos, webcam (realtime), and rtsp (realtime)
