# ============================================================
# IMPORT LIBRARY
# ============================================================
import streamlit as st
import cv2
import face_recognition
import pickle
import numpy as np
from ultralytics import YOLO
from datetime import datetime, time, timedelta
import pandas as pd
import plotly.express as px
import os
from nama_mahasiswa import NAMA_LENGKAP

st.set_page_config(
    page_title="Presensi ULBI",
    page_icon="logo_ulbi.png",
    layout="wide"
)


# ============================================================
# KONFIGURASI MATA KULIAH
# ============================================================
MATKUL = "Rekayasa & Desain Sistem Logistik"
KELAS = "4A - S1 Sains Data"
DOSEN = "Fatia Amalia Maresti, S.Si., M.Si."
JAM_MULAI = time(13, 30)
TOLERANSI_MENIT = 10

# Nama (sesuai NAMA_LENGKAP) dari subjek yang berperan sebagai dosen.
# Dipakai untuk mengecualikan dosen dari statistik kehadiran mahasiswa
# di Rekap Presensi Hari Ini dan Tab Analisis Data Presensi.
NAMA_DOSEN = "Fatia Amalia Maresti, S.Si., M.Si."

# Daftar nama mahasiswa saja (tanpa dosen), dipakai untuk statistik
NAMA_MAHASISWA = [nama for nama in NAMA_LENGKAP.values() if nama != NAMA_DOSEN]

# ============================================================
# HELPER: FORMAT TANGGAL BAHASA INDONESIA
# ============================================================
HARI_ID = {
    "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
    "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"
}
BULAN_ID = {
    "January": "Januari", "February": "Februari", "March": "Maret",
    "April": "April", "May": "Mei", "June": "Juni", "July": "Juli",
    "August": "Agustus", "September": "September", "October": "Oktober",
    "November": "November", "December": "Desember"
}

# Nama bulan singkat (3 huruf) berbahasa Indonesia, untuk penamaan file CSV
# Contoh: presensi-41-SD-jun26.csv -> Juni 2026
BULAN_SHORT_ID = {
    "January": "jan", "February": "feb", "March": "mar",
    "April": "apr", "May": "mei", "June": "jun", "July": "jul",
    "August": "agu", "September": "sep", "October": "okt",
    "November": "nov", "December": "des"
}

def format_tanggal_id(dt):
    hari = HARI_ID[dt.strftime("%A")]
    bulan = BULAN_ID[dt.strftime("%B")]
    return f"{hari}, {dt.strftime('%d')} {bulan} {dt.strftime('%Y')}"


# ============================================================
# HELPER: CEK STATUS KEHADIRAN
# ============================================================
def cek_status(waktu_scan):
    batas = datetime.combine(waktu_scan.date(), JAM_MULAI)
    batas_telat = batas + timedelta(minutes=TOLERANSI_MENIT)
    if waktu_scan <= batas_telat:
        return "Hadir"
    else:
        selisih = int((waktu_scan - batas_telat).total_seconds() / 60)
        if selisih >= 60:
            jam = selisih // 60
            menit = selisih % 60
            if menit == 0:
                return f"Terlambat {jam} jam"
            else:
                return f"Terlambat {jam} jam {menit} menit"
        else:
            return f"Terlambat {selisih} menit"


# ============================================================
# HELPER: LOAD & SIMPAN PRESENSI CSV
# ============================================================
def get_csv_path(dt=None):
    """
    Menghasilkan path file presensi berdasarkan bulan & tahun.
    Format: presensi-41-SD-<bulanpendek><2digittahun>.csv
    Contoh: presensi-41-SD-jun26.csv -> Juni 2026
    Default: bulan & tahun saat ini.
    """
    if dt is None:
        dt = datetime.now()
    bulan_pendek = BULAN_SHORT_ID[dt.strftime("%B")]
    tahun_pendek = dt.strftime("%y")
    return f"presensi-41-SD-{bulan_pendek}{tahun_pendek}.csv"

def load_presensi():
    path = get_csv_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Foto" not in df.columns:
            df["Foto"] = ""
        return df
    else:
        return pd.DataFrame(columns=[
            "Tanggal", "Nama", "Mata Kuliah", "Kelas",
            "Dosen", "Jam Mulai", "Waktu Absen", "Status", "Foto"
        ])

def simpan_presensi(df):
    df.to_csv(get_csv_path(), index=False)


def load_semua_presensi(folder="."):
    """
    Membaca dan menggabungkan semua file presensi-41-SD-*.csv yang ditemukan
    di folder, lalu mengembalikan satu DataFrame gabungan.

    File yang gagal dibaca (rusak/format tidak sesuai) akan dilewati.
    Kolom 'Tanggal' diparsing jadi datetime untuk keperluan filter rentang.
    """
    kolom_standar = [
        "Tanggal", "Nama", "Mata Kuliah", "Kelas",
        "Dosen", "Jam Mulai", "Waktu Absen", "Status"
    ]

    daftar_df = []
    for nama_file in sorted(os.listdir(folder)):
        if nama_file.startswith("presensi-41-SD-") and nama_file.endswith(".csv"):
            try:
                df_bulan = pd.read_csv(os.path.join(folder, nama_file))
                # Pastikan kolom yang diharapkan ada, lewati file yang formatnya tidak sesuai
                if all(kol in df_bulan.columns for kol in kolom_standar):
                    daftar_df.append(df_bulan)
            except Exception:
                continue

    if not daftar_df:
        return pd.DataFrame(columns=kolom_standar + ["Tanggal_dt"])

    df_gabungan = pd.concat(daftar_df, ignore_index=True)
    df_gabungan["Tanggal_dt"] = pd.to_datetime(
        df_gabungan["Tanggal"], format="%d-%m-%Y", errors="coerce"
    )
    df_gabungan = df_gabungan.dropna(subset=["Tanggal_dt"])
    df_gabungan = df_gabungan.sort_values("Tanggal_dt").reset_index(drop=True)
    return df_gabungan


# ============================================================
# LOAD MODEL YOLO DAN DATABASE ENCODING
# ============================================================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

@st.cache_resource
def load_model_spoofing():
    """
    Model YOLOv8s hasil training sendiri pada dataset gabungan
    4 dataset Roboflow: anti_spoofing (kelas device), anti_counterfeiting
    (HANDPHONE, LAPTOP, TABLET), anti_spoofing_v2 (Handphone), 50gambarhp
    (handphone). Total 1251 gambar, 1 kelas: spoofing (HP, laptop, tablet).
    mAP@50 pada test set: 0.9950.
    """
    model_spoofing = YOLO("best_spoofing.pt")
    return model_spoofing

@st.cache_resource
def load_encoding():
    with open("database_encoding.pkl", "rb") as f:
        data = pickle.load(f)
    names = []
    encodings = []
    for nama, enc_list in data.items():
        for enc in enc_list:
            names.append(nama)
            encodings.append(enc)
    return names, encodings

model = load_model()
model_spoofing = load_model_spoofing()
names, encodings = load_encoding()

# ============================================================
# FOLDER PENYIMPANAN FOTO DETEKSI
# ============================================================
FOTO_DIR = "deteksi_foto"
os.makedirs(FOTO_DIR, exist_ok=True)

# ============================================================
# MITIGASI SPOOFING: DETEKSI PERANGKAT ELEKTRONIK
# ============================================================
# Model deteksi spoofing (best_spoofing.pt) dilatih sendiri menggunakan
# dataset gabungan 4 dataset Roboflow (total 1251 gambar, 1 kelas: spoofing).
# Kelas spoofing mencakup HP, laptop, dan tablet.
# mAP@50 pada test set: 0.9950.
#
# Catatan scope (Bab V - Limitasi):
# Pendekatan ini HANYA menutup celah spoofing via perangkat elektronik.
# Foto cetak/kertas atau kartu ID tanpa perangkat tetap tidak terdeteksi
# oleh mekanisme ini, karena tidak ada objek "spoofing" yang terdeteksi.
SPOOFING_CLASS_ID = 0   # satu-satunya kelas di model custom: 'spoofing'
SPOOFING_CONF_THRESHOLD = 0.6
SPOOFING_OVERLAP_MIN = 0.8


def hitung_overlap_ratio(box_a, box_b):
    """
    Menghitung rasio area irisan (intersection) terhadap luas box_a.
    box_a, box_b: (x1, y1, x2, y2).
    Dipakai untuk cek apakah bbox perangkat spoofing berdekatan/tumpang
    tindih dengan bbox wajah, supaya perangkat yang kebetulan ada di
    background tidak ikut menolak presensi.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    luas_irisan = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    luas_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return luas_irisan / luas_a


def deteksi_spoofing_di_frame(frame_bgr, box_wajah, model_spoofing):
    """
    Menjalankan model spoofing (best_spoofing.pt) untuk mencari objek
    'spoofing' (HP, laptop, tablet) di frame, lalu cek apakah objek
    tersebut berdekatan dengan bbox wajah.
    Bbox yang menutupi lebih dari 80% area frame diabaikan karena
    kemungkinan besar adalah layar perangkat itu sendiri, bukan
    perangkat yang dipegang di depan wajah.
    Mengembalikan (True/False, confidence atau None, box koordinat atau None).
    """
    hasil = model_spoofing(frame_bgr, conf=SPOOFING_CONF_THRESHOLD, verbose=False)
    boxes = hasil[0].boxes

    frame_h, frame_w = frame_bgr.shape[:2]
    frame_area = frame_h * frame_w

    spoofing_terdeteksi = False
    spoofing_conf_final = None
    spoofing_box_final  = None

    for box in boxes:
        cls_id  = int(box.cls[0].cpu().numpy())
        conf    = float(box.conf[0].cpu().numpy())
        box_obj = box.xyxy[0].cpu().numpy().astype(int)
        overlap = hitung_overlap_ratio(box_wajah, box_obj)

        # Hitung rasio luas bbox spoofing terhadap seluruh frame
        bx1, by1, bx2, by2 = box_obj[0], box_obj[1], box_obj[2], box_obj[3]
        box_area = (bx2 - bx1) * (by2 - by1)
        rasio_frame = box_area / frame_area

        # Skip kalau bbox terlalu besar (kemungkinan layar perangkat itu sendiri)
        if rasio_frame > 0.8:
            continue

        if cls_id == SPOOFING_CLASS_ID and overlap >= SPOOFING_OVERLAP_MIN:
            spoofing_terdeteksi = True
            spoofing_conf_final = conf
            spoofing_box_final  = box_obj

    return spoofing_terdeteksi, spoofing_conf_final, spoofing_box_final


def simpan_foto_deteksi(frame_bgr, detections, nama_lengkap, yolo_conf, jarak_encoding, waktu_scan, spoofing_box=None):
    """
    Menggambar bounding box + label ke frame asli (BGR) lalu menyimpannya.
    Mengembalikan path file yang disimpan.

    Menggambar SEMUA box di `detections`, bukan cuma box pertama. Ini penting
    untuk kasus penolakan multi-wajah, supaya foto bukti menunjukkan persis
    berapa banyak & di mana wajah yang terdeteksi YOLO.

    Jika spoofing_box disediakan, gambar bbox biru di area perangkat spoofing.

    Label format (1 wajah): "Nama Lengkap | YOLO: 0.91 | dist: 0.312"
    Label format (>1 wajah): "Nama Lengkap #1 | YOLO: 0.91" (per box)
    Bounding box hijau jika dikenali, merah jika ditolak/tidak dikenali.
    """
    frame_out = frame_bgr.copy()

    if nama_lengkap == "Unknown" or nama_lengkap.startswith("Ditolak"):
        warna = (0, 0, 220)
    else:
        warna = (0, 200, 80)

    jumlah_box = len(detections)
    for i in range(jumlah_box):
        box = detections[i].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        conf_box = float(detections[i].conf[0].cpu().numpy())

        if jumlah_box == 1:
            label = f"{nama_lengkap} | YOLO: {yolo_conf:.2f} | dist: {jarak_encoding:.3f}"
        else:
            label = f"{nama_lengkap} #{i + 1} | YOLO: {conf_box:.2f}"

        cv2.rectangle(frame_out, (x1, y1), (x2, y2), warna, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = y1 - 8 if y1 - 8 > th else y1 + th + 8
        cv2.rectangle(frame_out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), warna, -1)
        cv2.putText(frame_out, label, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Gambar bbox perangkat spoofing (biru) jika ada
    if spoofing_box is not None:
        sx1, sy1, sx2, sy2 = spoofing_box[0], spoofing_box[1], spoofing_box[2], spoofing_box[3]
        cv2.rectangle(frame_out, (sx1, sy1), (sx2, sy2), (255, 100, 0), 2)
        slabel = "spoofing"
        (stw, sth), _ = cv2.getTextSize(slabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        slabel_y = sy1 - 8 if sy1 - 8 > sth else sy1 + sth + 8
        cv2.rectangle(frame_out, (sx1, slabel_y - sth - 4), (sx1 + stw + 4, slabel_y + 2), (255, 100, 0), -1)
        cv2.putText(frame_out, slabel, (sx1 + 2, slabel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    nama_file = nama_lengkap.lower().replace(" ", "_")
    timestamp = waktu_scan.strftime("%Y-%m-%d_%H-%M-%S")
    path_foto = os.path.join(FOTO_DIR, f"{nama_file}_{timestamp}.jpg")
    cv2.imwrite(path_foto, frame_out)
    return path_foto


# ============================================================
# SIDEBAR: NAVIGASI
# ============================================================
with st.sidebar:
    st.image("logo_ulbi.png", width=80)
    st.markdown("### Presensi ULBI")
    st.caption("Program Studi S1 Sains Data")
    st.divider()

    halaman = st.radio(
        "Menu",
        ["📷 Presensi", "📊 Analisis Data Presensi", "🔍 Performa Sistem"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(f"{KELAS}")
    st.caption(f"{DOSEN}")


# ============================================================
# HEADER (info matkul + jam, tampil di semua halaman)
# ============================================================
if halaman == "📷 Presensi":
    with st.container(border=True):
        st.subheader(MATKUL)

        col_kelas, col_dosen, col_jam, col_waktu = st.columns(4)
        with col_kelas:
            st.caption("🏷️ Kelas")
            st.markdown(f"**{KELAS}**")
        with col_dosen:
            st.caption("👤 Dosen")
            st.markdown(f"**{DOSEN}**")
        with col_jam:
            st.caption("🕐 Jam Kuliah")
            st.markdown(f"**Kamis, {JAM_MULAI.strftime('%H:%M')} - 15:30**")
        with col_waktu:
            sekarang = datetime.now()
            st.caption("🗓️ Waktu Sekarang")
            st.markdown(
                f"<p style='font-size:32px;font-weight:600;margin:0;line-height:1;font-variant-numeric:tabular-nums;letter-spacing:-1px;'>{sekarang.strftime('%H:%M:%S')}</p><p style='font-size:12px;color:gray;margin:2px 0 0;'>{format_tanggal_id(sekarang)}</p>",
                unsafe_allow_html=True
            )

    st.write("")

# ============================================================
# HALAMAN 1: PRESENSI
# ============================================================
if halaman == "📷 Presensi":

    col_kiri, col_kanan = st.columns([3, 2], gap="medium")

    # --------------------------------------------------------
    # KOLOM KIRI: KAMERA + MITIGASI SPOOFING (DETEKSI PERANGKAT) + HASIL DETEKSI
    # --------------------------------------------------------
    with col_kiri:
        st.subheader("Face Recognition Attendance System")
        st.caption("Arahkan wajah ke kamera dan pastikan tidak ada penghalang di depan wajah.")

        foto_input = st.camera_input("Ambil Foto Presensi", label_visibility="collapsed")

        if foto_input is not None:
            bytes_data = foto_input.getvalue()
            img_array = np.frombuffer(bytes_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

            with st.spinner("Memproses presensi..."):
                results = model(frame)
                detections = results[0].boxes

            if len(detections) == 0:
                st.warning("Wajah tidak terdeteksi. Coba lagi dengan posisi wajah yang lebih jelas.")
                waktu_scan = datetime.now()
                simpan_foto_deteksi(
                    frame, detections, "Ditolak (Wajah Tidak Terdeteksi)", 0.0, -1.0, waktu_scan
                )
            elif len(detections) > 1:
                st.error("Terdeteksi lebih dari satu wajah dalam frame. Pastikan hanya wajah Anda sendiri yang terlihat saat presensi.")
                waktu_scan = datetime.now()
                simpan_foto_deteksi(
                    frame, detections, "Ditolak (Multi-Wajah)", 0.0, -1.0, waktu_scan
                )
            else:
                box_wajah = detections[0].xyxy[0].cpu().numpy().astype(int)

                # ----------------------------------------------------
                # MITIGASI SPOOFING: cek dulu apakah ada perangkat spoofing
                # di area wajah sebelum lanjut ke face recognition. Lihat
                # fungsi deteksi_spoofing_di_frame() di bagian atas file.
                # ----------------------------------------------------
                spoofing_terdeteksi, spoofing_conf, spoofing_box = deteksi_spoofing_di_frame(frame, box_wajah, model_spoofing)

                if spoofing_terdeteksi:
                    st.error(
                        "🚫 Presensi ditolak. Terdeteksi perangkat elektronik (HP/laptop/tablet) "
                        "di depan wajah. Harap lakukan presensi secara langsung di depan kamera."
                    )
                    waktu_scan = datetime.now()
                    tanggal = waktu_scan.strftime("%d-%m-%Y")
                    waktu_str = waktu_scan.strftime("%H:%M:%S")
                    path_foto = simpan_foto_deteksi(
                        frame, detections, "Ditolak (Terdeteksi Spoofing)",
                        float(detections[0].conf[0].cpu().numpy()), -1.0, waktu_scan,
                        spoofing_box=spoofing_box
                    )
                    baris_baru = {
                        "Tanggal": tanggal,
                        "Nama": "Ditolak (Terdeteksi Spoofing)",
                        "Mata Kuliah": MATKUL,
                        "Kelas": KELAS,
                        "Dosen": DOSEN,
                        "Jam Mulai": JAM_MULAI.strftime("%H:%M"),
                        "Waktu Absen": waktu_str,
                        "Status": "Ditolak",
                        "Foto": path_foto
                    }
                    df = load_presensi()
                    df = pd.concat([df, pd.DataFrame([baris_baru])], ignore_index=True)
                    simpan_presensi(df)
                else:
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    if len(face_encodings) == 0:
                        st.warning("Wajah terdeteksi YOLO tapi encoding gagal. Coba lagi.")
                    else:
                        encoding = face_encodings[0]

                        matches = face_recognition.compare_faces(encodings, encoding, tolerance=0.5)
                        distances = face_recognition.face_distance(encodings, encoding)
                        best_idx = np.argmin(distances)

                        if matches[best_idx]:
                            nama_panggilan = names[best_idx]
                            nama_lengkap = NAMA_LENGKAP.get(nama_panggilan, nama_panggilan)
                            waktu_scan = datetime.now()
                            tanggal = waktu_scan.strftime("%d-%m-%Y")
                            waktu_str = waktu_scan.strftime("%H:%M:%S")

                            yolo_conf = float(detections[0].conf[0].cpu().numpy())
                            jarak_encoding = float(distances[best_idx])

                            df = load_presensi()

                            sudah_absen = (
                                (df["Nama"] == nama_lengkap) &
                                (df["Tanggal"] == tanggal)
                            ).any()

                            if sudah_absen:
                                waktu_pertama = df.loc[
                                    (df["Nama"] == nama_lengkap) & (df["Tanggal"] == tanggal),
                                    "Waktu Absen"
                                ].values[0]
                                st.info(f"ℹ️ {nama_lengkap} sudah tercatat hadir pukul {waktu_pertama}")
                            else:
                                status = cek_status(waktu_scan)
                                path_foto = simpan_foto_deteksi(
                                    frame, detections, nama_lengkap,
                                    yolo_conf, jarak_encoding, waktu_scan
                                )
                                baris_baru = {
                                    "Tanggal": tanggal,
                                    "Nama": nama_lengkap,
                                    "Mata Kuliah": MATKUL,
                                    "Kelas": KELAS,
                                    "Dosen": DOSEN,
                                    "Jam Mulai": "13:30",
                                    "Waktu Absen": waktu_str,
                                    "Status": status,
                                    "Foto": path_foto
                                }
                                df = pd.concat([df, pd.DataFrame([baris_baru])], ignore_index=True)
                                simpan_presensi(df)

                                if status == "Hadir":
                                    st.success(f"✅ {nama_lengkap} — {status}")
                                else:
                                    st.warning(f"🕐 {nama_lengkap} — {status}")
                                st.caption(f"{format_tanggal_id(waktu_scan)}, pukul {waktu_str}")
                        else:
                            waktu_scan = datetime.now()
                            yolo_conf = float(detections[0].conf[0].cpu().numpy())
                            jarak_encoding = float(distances[best_idx])
                            simpan_foto_deteksi(
                                frame, detections, "Unknown",
                                yolo_conf, jarak_encoding, waktu_scan
                            )
                            st.error("Wajah tidak dikenali dalam database.")

    # --------------------------------------------------------
    # KOLOM KANAN: REKAP HARI INI
    # --------------------------------------------------------
    with col_kanan:
        st.subheader("Sudah Absen Hari Ini")

        df = load_presensi()
        tanggal_hari_ini = datetime.now().strftime("%d-%m-%Y")
        df_hari_ini = df[df["Tanggal"] == tanggal_hari_ini]

        def beri_badge(status):
            if status == "Hadir":
                return "🟢 Hadir"
            elif status.startswith("Terlambat"):
                return f"🟠 {status}"
            return status

        df_dosen_hari_ini = df_hari_ini[df_hari_ini["Nama"] == NAMA_DOSEN]
        df_mhs_hari_ini = df_hari_ini[df_hari_ini["Nama"].isin(NAMA_MAHASISWA)]

        # ── Info dosen ──
        if not df_dosen_hari_ini.empty:
            baris_dosen = df_dosen_hari_ini.iloc[0]
            st.caption(
                f"👤 Dosen: {NAMA_DOSEN} — "
                f"{beri_badge(baris_dosen['Status'])} pukul {baris_dosen['Waktu Absen']}"
            )
        else:
            st.caption(f"👤 Dosen: {NAMA_DOSEN} — belum presensi")

        # ── 3 Metric card ──
        jumlah_hadir = len(df_mhs_hari_ini)
        total_mhs = len(NAMA_MAHASISWA)
        belum_absen = total_mhs - jumlah_hadir
        pct_hadir = round(jumlah_hadir / total_mhs * 100) if total_mhs > 0 else 0

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🟢 Hadir", jumlah_hadir, help=f"Dari {total_mhs} mahasiswa")
        mc2.metric("⏳ Belum Absen", belum_absen)
        mc3.metric("📊 Kehadiran", f"{pct_hadir}%")

        # ── Progress bar ──
        st.progress(
            jumlah_hadir / total_mhs if total_mhs > 0 else 0,
            text=f"{jumlah_hadir} dari {total_mhs} mahasiswa sudah presensi"
        )

        # ── Tabel rekap ──
        if df_mhs_hari_ini.empty:
            st.info("Belum ada mahasiswa yang presensi hari ini.")
        else:
            df_tampil = df_mhs_hari_ini[["Nama", "Waktu Absen", "Status"]].copy()
            df_tampil["Status"] = df_tampil["Status"].apply(beri_badge)
            st.dataframe(
                df_tampil,
                use_container_width=True,
                hide_index=True,
                height=min(400, 35 * (len(df_tampil) + 1) + 10)
            )


# ============================================================
# HALAMAN 2: ANALISIS DATA PRESENSI
# ============================================================
elif halaman == "📊 Analisis Data Presensi":
    st.subheader("Analisis Data Presensi")

    df_all = load_semua_presensi()

    if df_all.empty:
        st.info("Belum ada data presensi yang bisa dianalisis.")
    else:
        tanggal_min = df_all["Tanggal_dt"].min().date()
        tanggal_max = df_all["Tanggal_dt"].max().date()

        rentang = st.date_input(
            "Pilih rentang tanggal",
            value=(tanggal_min, tanggal_max),
            min_value=tanggal_min,
            max_value=tanggal_max,
        )

        if isinstance(rentang, tuple) and len(rentang) == 2:
            mulai, akhir = rentang
        else:
            mulai, akhir = tanggal_min, tanggal_max

        mask = (
            (df_all["Tanggal_dt"].dt.date >= mulai) &
            (df_all["Tanggal_dt"].dt.date <= akhir)
        )
        df_range = df_all[mask].copy()

        if df_range.empty:
            st.warning("Tidak ada data presensi pada rentang tanggal yang dipilih.")
        else:
            tanggal_pertemuan = sorted(df_range["Tanggal"].unique())
            semua_nama = NAMA_MAHASISWA
            total_pertemuan = len(tanggal_pertemuan)

            # Hitung rekap per (mahasiswa, tanggal): Hadir / Terlambat / Alpa
            rekap_rows = []
            for tgl in tanggal_pertemuan:
                df_tgl = df_range[df_range["Tanggal"] == tgl]
                for nama in semua_nama:
                    row = df_tgl[df_tgl["Nama"] == nama]
                    if row.empty:
                        status_kategori = "Alpa"
                    elif row.iloc[0]["Status"].startswith("Terlambat"):
                        status_kategori = "Terlambat"
                    else:
                        status_kategori = "Hadir"
                    rekap_rows.append({"Tanggal": tgl, "Nama": nama, "Status": status_kategori})

            df_rekap = pd.DataFrame(rekap_rows)

            # Rekap per mahasiswa
            rekap_mhs = df_rekap.groupby(["Nama", "Status"]).size().unstack(fill_value=0)
            for kolom in ["Hadir", "Terlambat", "Alpa"]:
                if kolom not in rekap_mhs.columns:
                    rekap_mhs[kolom] = 0
            rekap_mhs["Total Pertemuan"] = total_pertemuan
            rekap_mhs["% Kehadiran"] = (
                (rekap_mhs["Hadir"] + rekap_mhs["Terlambat"]) / total_pertemuan * 100
            ).round(1)
            rekap_mhs = rekap_mhs.sort_values("% Kehadiran", ascending=False).reset_index()
            rekap_mhs = rekap_mhs[["Nama", "Hadir", "Terlambat", "Alpa", "Total Pertemuan", "% Kehadiran"]]

            # Hitung summary
            rata_kehadiran = rekap_mhs["% Kehadiran"].mean().round(1)
            selalu_hadir = len(rekap_mhs[rekap_mhs["% Kehadiran"] == 100])
            perlu_perhatian = len(rekap_mhs[rekap_mhs["% Kehadiran"] < 75])

            # --------------------------------------------------------
            # SUMMARY CARD
            # --------------------------------------------------------
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Pertemuan", total_pertemuan)
            c2.metric("Rata-rata Kehadiran", f"{rata_kehadiran}%")
            c3.metric("Selalu Hadir", f"{selalu_hadir} mhs")
            c4.metric("Perlu Perhatian", f"{perlu_perhatian} mhs",
                      help="Mahasiswa dengan kehadiran di bawah 75%")

            st.divider()

            # --------------------------------------------------------
            # PODIUM 3 MAHASISWA TERAJIN
            # --------------------------------------------------------
            top3 = rekap_mhs.sort_values("% Kehadiran", ascending=False).head(3).reset_index(drop=True)

            def get_inisial(nama):
                parts = nama.split()
                if len(parts) >= 2:
                    return parts[0][0].upper() + parts[1][0].upper()
                return parts[0][:2].upper()

            def podium_card(row, tinggi, ukuran_avatar, warna_avatar, warna_blok, warna_angka, warna_pct):
                inisial = get_inisial(row["Nama"])
                return f"""
                <div style="display:flex;flex-direction:column;align-items:center;gap:6px;width:180px">
                    <div style="width:{ukuran_avatar}px;height:{ukuran_avatar}px;border-radius:50%;background:{warna_avatar};display:flex;align-items:center;justify-content:center;font-size:{ukuran_avatar//3}px;font-weight:500;color:#fff">{inisial}</div>
                    <div style="font-size:12px;font-weight:500;color:var(--color-text-primary);text-align:center;line-height:1.4">{row['Nama']}</div>
                    <div style="font-size:13px;font-weight:500;color:{warna_pct}">{row['% Kehadiran']}%</div>
                    <div style="width:100%;height:{tinggi}px;border-radius:8px 8px 0 0;background:{warna_blok};display:flex;align-items:center;justify-content:center">
                        <span style="font-size:{tinggi//4}px;font-weight:500;color:{warna_angka}">{row.name + 1}</span>
                    </div>
                </div>
                """

            if len(top3) >= 3:
                st.markdown("#### Mahasiswa Terajin")
                st.caption(f"Berdasarkan persentase kehadiran tertinggi · {mulai.strftime('%d %b %Y')} - {akhir.strftime('%d %b %Y')}")

                html_podium = f"""
                <div style="display:flex;align-items:flex-end;justify-content:center;gap:12px;margin-bottom:1rem">
                    {podium_card(top3.iloc[1], 80, 44, "#888780", "#B4B2A9", "#444441", "#5F5E5A")}
                    {podium_card(top3.iloc[0], 110, 52, "#EF9F27", "#FAC775", "#633806", "#854F0B")}
                    {podium_card(top3.iloc[2], 60, 40, "#97C459", "#C0DD97", "#27500A", "#3B6D11")}
                </div>
                """
                st.markdown(html_podium, unsafe_allow_html=True)

            st.divider()

            # --------------------------------------------------------
            # BARIS 1: DONUT + PERLU PERHATIAN
            # --------------------------------------------------------
            col_donut, col_perhatian = st.columns([1, 1], gap="medium")

            with col_donut:
                st.markdown("#### Ringkasan Kehadiran Kelas")
                st.caption(f"{total_pertemuan} pertemuan · {mulai.strftime('%d %b %Y')} - {akhir.strftime('%d %b %Y')}")
                ringkasan = df_rekap["Status"].value_counts().reset_index()
                ringkasan.columns = ["Status", "Jumlah"]
                warna_status = {"Hadir": "#2E7D32", "Terlambat": "#F9A825", "Alpa": "#C62828"}
                fig_donut = px.pie(
                    ringkasan, names="Status", values="Jumlah",
                    hole=0.5, color="Status", color_discrete_map=warna_status
                )
                fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_perhatian:
                st.markdown("#### Mahasiswa Perlu Perhatian")
                st.caption("Kehadiran di bawah 75%")
                df_bawah = rekap_mhs[rekap_mhs["% Kehadiran"] < 75][["Nama", "% Kehadiran", "Alpa"]].copy()
                if df_bawah.empty:
                    st.success("Semua mahasiswa memiliki kehadiran di atas 75%.")
                else:
                    df_bawah = df_bawah.sort_values("% Kehadiran")
                    st.dataframe(df_bawah, use_container_width=True, hide_index=True, height=300)

            st.divider()

            # --------------------------------------------------------
            # BARIS 2: REKAP PER MAHASISWA + BAR CHART
            # --------------------------------------------------------
            st.markdown("#### Rekap Kehadiran per Mahasiswa")

            col_tabel, col_bar = st.columns([1, 1], gap="medium")

            with col_bar:
                fig_bar = px.bar(
                    rekap_mhs.sort_values("% Kehadiran", ascending=True),
                    x="% Kehadiran", y="Nama", orientation="h",
                    range_x=[0, 100],
                    color="% Kehadiran",
                    color_continuous_scale=["#C62828", "#F9A825", "#2E7D32"],
                    labels={"% Kehadiran": "% Kehadiran", "Nama": ""},
                )
                fig_bar.update_layout(
                    height=max(400, 28 * len(rekap_mhs)),
                    margin=dict(t=10, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_tabel:
                st.dataframe(
                    rekap_mhs.sort_values("Nama").reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=max(400, 28 * len(rekap_mhs))
                )

            st.divider()

            # --------------------------------------------------------
            # DOWNLOAD CSV (sesuai rentang tanggal)
            # --------------------------------------------------------
            st.markdown("#### Unduh Data Presensi")
            st.caption(f"Data yang diunduh sesuai rentang tanggal yang dipilih: {mulai.strftime('%d %b %Y')} - {akhir.strftime('%d %b %Y')}")

            df_download = df_range[df_range["Nama"] != NAMA_DOSEN].copy()
            kolom_download = ["Tanggal", "Nama", "Waktu Absen", "Status"]
            df_download = df_download[kolom_download]

            csv_bytes = df_download.to_csv(index=False).encode("utf-8")
            nama_file_unduh = f"presensi-4A-SD_{mulai.strftime('%d%m%Y')}-{akhir.strftime('%d%m%Y')}.csv"

            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=nama_file_unduh,
                mime="text/csv"
            )

# ============================================================
# HALAMAN 3: PERFORMA SISTEM
# ============================================================
elif halaman == "🔍 Performa Sistem":
    st.subheader("Performa Sistem")

    # --------------------------------------------------------
    # INFO MODEL (hardcode — statis)
    # --------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model Deteksi", "YOLOv8s")
    col_m2.metric("Library Recognition", "dlib ResNet-34")
    col_m3.metric("Subjek Terdaftar", f"{len(NAMA_LENGKAP)} orang")
    total_encoding = len(encodings)
    col_m4.metric("Total Encoding", f"{total_encoding:,}")

    st.divider()

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Model YOLOv8s", "🧑 Face Recognition", "📋 Log Deteksi"])

    # --------------------------------------------------------
    # TAB 1: MODEL YOLOV8S
    # --------------------------------------------------------
    with tab1:
        st.markdown("#### Performa Training Model YOLOv8")
        st.caption(
            "mAP@50 (mean Average Precision pada IoU 0,5) ditetapkan dengan threshold 0,5 "
            "karena mempertimbangkan kebutuhan deteksi wajah yang cukup akurat untuk tahap "
            "face recognition selanjutnya."
        )

        def render_training_chart(results_path, nama_model, map_test):
            if not os.path.exists(results_path):
                st.info(f"File {results_path} tidak ditemukan.")
                return

            df_results = pd.read_csv(results_path)
            df_results.columns = df_results.columns.str.strip()

            col_map_col = None
            for c in df_results.columns:
                if "map50" in c.lower() and "95" not in c.lower():
                    col_map_col = c
                    break

            if not col_map_col:
                st.info(f"Kolom mAP tidak ditemukan di {results_path}.")
                return

            best_idx = df_results[col_map_col].idxmax()
            best_epoch = int(df_results.loc[best_idx, "epoch"])
            best_val_map = df_results.loc[best_idx, col_map_col]

            fig = px.line(
                df_results, x="epoch", y=col_map_col,
                markers=False,
                labels={"epoch": "Epoch", col_map_col: "mAP@50 (Validasi)"}
            )
            fig.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            fig.update_traces(line_color="#2E7D32")
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("mAP@50 (Test Set)", f"{map_test:.4f}")
            mc2.metric(
                "Epoch Terbaik", best_epoch,
                help=f"mAP@50 validasi pada epoch ini: {best_val_map:.4f}"
            )

        col_face, col_spoofing = st.columns(2, gap="medium")
        with col_face:
            st.markdown("##### Model Deteksi Wajah")
            render_training_chart("results.csv", "Model Deteksi Wajah", 0.9372)
            st.caption("mAP@50 validasi per epoch. Metric card dari evaluasi test set (519 data).")
        with col_spoofing:
            st.markdown("##### Model Deteksi Spoofing")
            render_training_chart("results_spoofing.csv", "Model Deteksi Spoofing", 0.9950)
            st.caption("mAP@50 validasi per epoch. Metric card dari evaluasi test set (126 data).")

        st.divider()

        st.markdown("#### Confusion Matrix Model")

        def render_confusion_matrix(csv_path, nama_model, precision_val, recall_val):
            if not os.path.exists(csv_path):
                st.info(f"File {csv_path} tidak ditemukan.")
                return

            df_cm = pd.read_csv(csv_path, index_col=0)
            labels = df_cm.columns.tolist()

            fig = px.imshow(
                df_cm.values,
                x=labels,
                y=labels,
                color_continuous_scale="Blues",
                zmin=0,
                labels=dict(x="True", y="Predicted", color="Nilai")
            )
            fig.update_layout(
                title=nama_model,
                height=350,
                margin=dict(t=40, b=10, l=10, r=10),
                coloraxis_showscale=False
            )
            fig.update_traces(texttemplate="%{z:.2f}")
            st.plotly_chart(fig, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("Precision", f"{precision_val:.4f}")
            mc2.metric("Recall", f"{recall_val:.4f}")

        col_cm_face, col_cm_spoofing = st.columns(2, gap="medium")
        with col_cm_face:
            render_confusion_matrix("confusion_matrix_face.csv", "Model Deteksi Wajah", 0.9488, 0.8816)
            st.caption(
                "Confusion matrix dan metrik precision-recall dihitung pada 519 gambar test."
            )
        with col_cm_spoofing:
            render_confusion_matrix("confusion_matrix_spoofing.csv", "Model Deteksi Spoofing", 0.9971, 0.9921)
            st.caption(
                "Confusion matrix dan metrik precision-recall dihitung pada 126 gambar test."
            )

    # --------------------------------------------------------
    # TAB 2: FACE RECOGNITION
    # --------------------------------------------------------
    with tab2:
        st.markdown("#### Evaluasi Face Recognition")
        st.info("Bagian ini masih dalam proses pengerjaan.")
        st.markdown("""
        **Yang perlu dikerjakan:**
        - Lakukan pengujian pengenalan wajah untuk setiap subjek (minimal 5 kali per orang)
        - Catat hasil: benar dikenali, salah dikenali, atau tidak dikenali
        - Hitung metrik evaluasi:
            - **Accuracy** — persentase identitas yang dikenali benar
            - **FAR** (False Accept Rate) — orang lain dikenali sebagai subjek terdaftar
            - **FRR** (False Reject Rate) — subjek terdaftar tidak dikenali sistem
        """)

    # --------------------------------------------------------
    # TAB 3: LOG DETEKSI & FOTO PRESENSI
    # --------------------------------------------------------
    with tab3:
        st.markdown("#### Log Deteksi & Foto Presensi")

        df_log = load_semua_presensi()
        df_log = df_log[df_log["Foto"].notna() & (df_log["Foto"] != "")]

        if df_log.empty:
            st.info("Belum ada log deteksi. Data akan muncul setelah mahasiswa melakukan presensi.")
        else:
            col_filter, col_info = st.columns([2, 1], gap="medium")

            with col_filter:
                pilihan_nama = ["Semua"] + sorted(df_log["Nama"].unique().tolist())
                nama_dipilih = st.selectbox("Pilih mahasiswa", pilihan_nama)

            if nama_dipilih == "Semua":
                df_tampil_log = df_log.copy()
            else:
                df_tampil_log = df_log[df_log["Nama"] == nama_dipilih].copy()

            df_tampil_log = df_tampil_log.sort_values("Waktu Absen", ascending=False)

            with col_info:
                st.metric("Total foto tersimpan", len(df_tampil_log))

            st.caption(f"Menampilkan {len(df_tampil_log)} entri")

            foto_list = df_tampil_log[["Nama", "Tanggal", "Waktu Absen", "Status", "Foto"]].values.tolist()

            if foto_list:
                cols = st.columns(3)
                for i, (nama, tanggal, waktu, status, path_foto) in enumerate(foto_list):
                    with cols[i % 3]:
                        if os.path.exists(str(path_foto)):
                            img_bgr = cv2.imread(path_foto)
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                        else:
                            st.warning("Foto tidak ditemukan")
                        st.caption(f"**{nama}**")
                        st.caption(f"{tanggal} · {waktu} · {status}")