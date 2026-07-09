import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="Dashboard Sentimen Konflik Knetz vs ASEAN",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS biar tampilannya lebih premium
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1e3d59;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff6e40;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi buat load dataset dengan caching
@st.cache_data
def load_data():
    file_path = "dataset.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        if 'cleaned_text' not in df.columns and 'normalized_text' in df.columns:
            df['cleaned_text'] = df['normalized_text']
        return df
    else:
        return pd.DataFrame({
            'full_text': ['contoh tweet 1', 'contoh tweet 2'],
            'cleaned_text': ['contoh tweet 1', 'contoh tweet 2'],
            'Kategori_Konflik': ['Netral / Informatif', 'Serangan / Hinaan']
        })

df = load_data()

# Rules labeling untuk fitur simulasi prediksi
labeling_rules = {
    'Serangan / Hinaan': ['serang', 'hina', 'bodoh', 'tolol', 'bego', 'sampah', 'goblok', 'babi', 'anjing', 'idiot'],
    'Rasisme': ['rasis', 'kulit', 'negara', 'miskin', 'kampungan', 'indonesia', 'korea', 'sipit', 'item', 'monyet'],
    'Agama / Budaya': ['agama', 'budaya', 'islam', 'hijab', 'halal', 'haram', 'tradisi', 'ibadah', 'tuhan'],
    'Pembelaan Diri': ['bela', 'salah', 'maaf', 'klarifikasi', 'fakta', 'benar', 'faktanya', 'jangan asal']
}

def predict_category(text):
    text = text.lower()
    for label, keywords in labeling_rules.items():
        for keyword in keywords:
            if keyword in text:
                return label
    return 'Netral / Informatif'


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3655/3655523.png", width=100)
    st.title("Menu Navigasi")
    menu = st.radio("Pilih Menu:", ["Ringkasan Data", "Analisis Sentimen", "Simulasi Model"])
    st.markdown("---")
    st.markdown("**Tugas Besar DIP**")
    st.markdown("Analisis Sentimen Konflik Knetz vs ASEAN.")


# ==========================
# MAIN CONTENT
# ==========================
if menu == "Ringkasan Data":
    st.title("📊 Ringkasan Dataset")
    st.markdown("Berikut adalah ringkasan dari data hasil *scraping* (Twitter) seputar konflik Knetz vs ASEAN yang telah dibersihkan.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Data Tersedia</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Baris Tweet</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        kategori_konflik = len(df[df['Kategori_Konflik'] != 'Netral / Informatif'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Terdeteksi Konflik / Sentimen Negatif</div>
            <div class="metric-value">{kategori_konflik}</div>
            <div class="metric-label">Baris Tweet</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Label Netral / Informatif</div>
            <div class="metric-value">{len(df) - kategori_konflik}</div>
            <div class="metric-label">Baris Tweet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Cuplikan Dataset")
    st.dataframe(df[['full_text', 'cleaned_text', 'Kategori_Konflik']].head(50), use_container_width=True)

elif menu == "Analisis Sentimen":
    st.title("📈 Visualisasi Kategori Konflik")
    st.markdown("Bagaimana distribusi sentimen konflik Knetz vs ASEAN di dalam dataset kita?")
    
    # Hitung jumlah per kategori
    distribusi = df['Kategori_Konflik'].value_counts().reset_index()
    distribusi.columns = ['Kategori', 'Jumlah']

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Bar Chart")
        fig_bar = px.bar(
            distribusi, 
            x='Kategori', 
            y='Jumlah',
            color='Kategori',
            text='Jumlah',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("Pie Chart (Non-Netral)")
        # Filter Netral biar keliatan chart signifikansinya
        distribusi_non_netral = distribusi[distribusi['Kategori'] != 'Netral / Informatif']
        if len(distribusi_non_netral) > 0:
            fig_pie = px.pie(
                distribusi_non_netral, 
                names='Kategori', 
                values='Jumlah',
                hole=0.4,
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Tidak ada data konflik untuk ditampilkan di Pie Chart.")

elif menu == "Simulasi Model":
    st.title("🤖 Simulasi Klasifikasi Tweet")
    st.markdown("Coba masukkan kalimat atau tweet terkait konflik di bawah ini, dan sistem akan memprediksi masuk ke sentimen apa kalimat tersebut.")
    
    user_input = st.text_area("Masukkan teks di sini:", placeholder="Contoh: Orang indonesia kalau dikasih tau pada baperan dan rasis banget...")
    
    if st.button("Prediksi Sekarang!", type="primary"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            prediksi = predict_category(user_input)
            st.success("Prediksi Berhasil!")
            
            st.markdown("### Hasil Prediksi:")
            if prediksi == "Netral / Informatif":
                st.info(f"Kategori: **{prediksi}** 🕊️ (Aman, tidak terdeteksi konflik)")
            elif prediksi == "Serangan / Hinaan":
                st.error(f"Kategori: **{prediksi}** 🤬")
            elif prediksi == "Rasisme":
                st.error(f"Kategori: **{prediksi}** 🚫")
            elif prediksi == "Agama / Budaya":
                st.warning(f"Kategori: **{prediksi}** 🕌")
            elif prediksi == "Pembelaan Diri":
                st.success(f"Kategori: **{prediksi}** 🛡️")
            else:
                st.success(f"Kategori: **{prediksi}**")
