import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Konfigurasi Halaman (Harus di paling atas)
st.set_page_config(
    page_title="Dashboard Sentimen & Kesehatan Mental",
    page_icon="🧠",
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

# Fungsi buat load dataset dengan caching biar kenceng
@st.cache_data
def load_data():
    # Pastikan file ada
    file_path = "dataset.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Handle kolom yang mungkin missing/beda nama
        if 'cleaned_text' not in df.columns and 'normalized_text' in df.columns:
            df['cleaned_text'] = df['normalized_text']
        return df
    else:
        # Fallback dummy data kalau error di HF Spaces (kalau user lupa upload excel)
        return pd.DataFrame({
            'full_text': ['contoh tweet 1', 'contoh tweet 2'],
            'cleaned_text': ['contoh tweet 1', 'contoh tweet 2'],
            'Kategori_Mental': ['Netral', 'Depresi']
        })

df = load_data()

# Rules labeling (sama dengan di Colab) untuk fitur simulasi prediksi
labeling_rules = {
    'Insomnia': ['tidur', 'lelah', 'begadang', 'ngantuk', 'insomnia', 'capek', 'melek'],
    'Depresi': ['sedih', 'nangis', 'menangis', 'putus asa', 'hancur', 'depresi', 'nyerah'],
    'Cemas': ['cemas', 'takut', 'khawatir', 'gugup', 'overthinking', 'panik', 'gelisah'],
    'Stress': ['stres', 'stress', 'pusing', 'muak', 'gila', 'beban', 'berat']
}

def predict_category(text):
    text = text.lower()
    for label, keywords in labeling_rules.items():
        for keyword in keywords:
            if keyword in text:
                return label
    return 'Lainnya / Netral'


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3655/3655523.png", width=100)
    st.title("Menu Navigasi")
    menu = st.radio("Pilih Menu:", ["Ringkasan Data", "Analisis Sentimen", "Simulasi Model"])
    st.markdown("---")
    st.markdown("**Tugas Besar DIP**")
    st.markdown("Analisis Sentimen Knetz vs ASEAN & Deteksi Keluhan Kesehatan Mental.")


# ==========================
# MAIN CONTENT
# ==========================
if menu == "Ringkasan Data":
    st.title("📊 Ringkasan Dataset")
    st.markdown("Berikut adalah ringkasan dari data hasil *scraping* yang telah dibersihkan.")

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
        kategori_keluhan = len(df[df['Kategori_Mental'] != 'Lainnya / Netral'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Terdeteksi Keluhan</div>
            <div class="metric-value">{kategori_keluhan}</div>
            <div class="metric-label">Baris Tweet</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Label Netral</div>
            <div class="metric-value">{len(df) - kategori_keluhan}</div>
            <div class="metric-label">Baris Tweet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Cuplikan Dataset")
    st.dataframe(df[['full_text', 'cleaned_text', 'Kategori_Mental']].head(50), use_container_width=True)

elif menu == "Analisis Sentimen":
    st.title("📈 Visualisasi Kategori Mental")
    st.markdown("Bagaimana distribusi keluhan yang ada di dalam dataset kita?")
    
    # Hitung jumlah per kategori
    distribusi = df['Kategori_Mental'].value_counts().reset_index()
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
        st.subheader("Pie Chart")
        # Filter Netral biar keliatan chart keluhannya
        distribusi_non_netral = distribusi[distribusi['Kategori'] != 'Lainnya / Netral']
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
            st.info("Tidak ada data keluhan untuk ditampilkan di Pie Chart.")

elif menu == "Simulasi Model":
    st.title("🤖 Simulasi Prediksi Kategori")
    st.markdown("Coba masukkan kalimat curhatan atau tweet di bawah ini, dan sistem akan memprediksi masuk ke kategori keluhan mental apa kalimat tersebut.")
    
    user_input = st.text_area("Masukkan teks di sini:", placeholder="Contoh: Aduh capek banget tugas numpuk rasanya pengen nyerah aja...")
    
    if st.button("Prediksi Sekarang!", type="primary"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong!")
        else:
            prediksi = predict_category(user_input)
            st.success("Prediksi Berhasil!")
            
            st.markdown("### Hasil Prediksi:")
            if prediksi == "Lainnya / Netral":
                st.info(f"Kategori: **{prediksi}** (Tidak terdeteksi keluhan)")
            elif prediksi == "Depresi":
                st.error(f"Kategori: **{prediksi}** 😢")
            elif prediksi == "Stress":
                st.warning(f"Kategori: **{prediksi}** 🤯")
            elif prediksi == "Cemas":
                st.warning(f"Kategori: **{prediksi}** 😰")
            elif prediksi == "Insomnia":
                st.info(f"Kategori: **{prediksi}** 🦉")
            else:
                st.success(f"Kategori: **{prediksi}**")
