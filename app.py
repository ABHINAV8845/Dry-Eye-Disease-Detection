import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Dry Eye Detector", page_icon="👁️", layout="centered")

# ---------- LOAD IMAGE ----------
try:
    eye_img = Image.open("eye.png")  # 👁️ Keep eye.png in same folder
except:
    eye_img = None

# ---------- BLUE + WHITE BACKGROUND STYLE ----------
page_bg = """
<style>
/* ===== Background ===== */
[data-testid="stAppViewContainer"] {
    background-color:#E3F2FD;  /* Light blue background */
    color: #000000;             /* Dark text for readability */
}

/* ===== Sidebar (optional) ===== */
[data-testid="stSidebar"] {
    background-color: #B71C1C;
}

/* ===== Headings ===== */
h1, h2, h3, h4 {
    color: #B71C1C;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Upload & Result Boxes ===== */
.upload-box {
    background-color: rgba(255,255,255,0.95);
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0, 123, 255, 0.2);
    padding: 2rem;
    margin-top: 20px;
}

.result-box {
    background-color: #ffffff;
    border: 2px solid #90caf9;
    border-radius: 12px;
    box-shadow: 0px 3px 8px rgba(33,150,243,0.2);
    padding: 1rem;
    margin-top: 1rem;
}

.advice {
    background-color: #bbdefb;
    border-radius: 10px;
    color: #0d47a1;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 1.05rem;
}

/* ===== Footer ===== */
footer {
    text-align: center;
    margin-top: 50px;
    color: #B71C1C;
    font-size: 0.9rem;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# ---------- HEADER ----------
if eye_img:
    st.image(eye_img, width=150)
st.markdown("<h1>👁️ Dry Eye Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#1565c0;'>"
    "Upload your gene expression file (CSV / TSV / TXT) to get your AI-based eye health report."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    clf = joblib.load("models/classifier.joblib")
    scaler = joblib.load("models/scaler.joblib")
    genes = joblib.load("models/selected_genes.joblib")
    return clf, scaler, genes

clf, scaler, genes = load_models()

# ---------- FILE UPLOAD ----------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📂 Upload your gene expression file", type=["csv", "tsv", "txt"])
st.markdown("</div>", unsafe_allow_html=True)

# ---------- MAIN LOGIC ----------
if uploaded:
    try:
        sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(uploaded, sep=sep)

        if df.shape[0] == 1:
            # Single-row sample: columns are genes, row is one sample
            s = df.iloc[0]
            s.index = s.index.astype(str)
        elif df.shape[1] >= 2:
            # First column: gene names, second column: expression values
            s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].astype(str).values)
        else:
            st.error("❌ Invalid file format. Please upload a valid CSV/TSV/TXT file.")
            st.stop()

        # Prepare sample
        s = s.reindex(genes).fillna(1e-6)
        X = scaler.transform([s])
        prob = clf.predict_proba(X)[0][1]
        label = "🩸 Dry Eye" if prob >= 0.5 else "🧿 No Dry Eye"
        confidence = round(prob * 100, 2)

        # ---------- RESULT ----------
        st.markdown("<div class='result-box'><h3>🔍 Prediction Result</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<h2 style='color:{'red' if prob >= 0.5 else 'green'}'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='text-align:center;'>Confidence: <b>{confidence}%</b></p>", unsafe_allow_html=True)
        st.progress(int(confidence))

        # ---------- HEALTH ADVICE ----------
        if prob >= 0.8:
            advice = "⚠️ High risk of Dry Eye. Please consult an eye specialist soon."
        elif 0.5 <= prob < 0.8:
            advice = "🧐 Mild risk of Dry Eye. Stay hydrated and follow the 20-20-20 eye rule."
        elif 0.35 <= prob < 0.5:
            advice = "🙂 Low risk. Keep taking care of your eyes and avoid excessive dryness."
        else:
            advice = "😊 Excellent! Your eyes seem healthy. Keep following good eye habits!"

        st.markdown(f"<div class='advice'><b>💡 Health Advice:</b> {advice}</div>", unsafe_allow_html=True)

        with st.expander("📊 View uploaded data preview"):
            st.dataframe(df.head())

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")

else:
    st.info("👆 Upload a file to start your AI health analysis.")

# ---------- FOOTER ----------
footer_html = """
<style>
.footer-box {
    background: linear-gradient(to right, #bbdefb, #e3f2fd);
    border-radius: 12px;
    padding: 15px 20px;
    margin-top: 40px;
    box-shadow: 0px 3px 8px rgba(33,150,243,0.3);
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
}
.footer-text {
    color: #0d47a1;
    font-size: 1rem;
    font-weight: 500;
    line-height: 1.6;
}
.footer-emoji {
    font-size: 1.1rem;
}
</style>

<div class="footer-box">
    <p class="footer-text">
        <span class="footer-emoji">💡👁️</span> <b>Eye Health Tip:</b><br>
        Spending long hours on screens 💻📱 makes you blink 👀 less often,<br>
        causing your tears 💧 to dry up faster.<br><br>
        This can lead to <b>Dry Eye</b> — irritation, redness, and tired eyes 😣.<br>
    </p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
