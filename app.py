import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_openml

# 1. ตั้งค่าหน้าเว็บ (ต้องเรียกเป็นคำสั่ง Streamlit คำสั่งแรก)
st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="centered")


# 2. โหลดโมเดลที่เซฟมาจาก Colab
@st.cache_resource
def load_model():
    return joblib.load("credit_pipeline.pkl")


# 3. โหลดโครงสร้างข้อมูลเพื่อเอามาทำฟอร์มรับค่า (cache ไว้ไม่ให้โหลดซ้ำทุกครั้ง)
@st.cache_data
def load_data():
    german_credit = fetch_openml(data_id=31, as_frame=True, parser="pandas")
    return german_credit.data


try:
    model = load_model()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ `credit_pipeline.pkl` กรุณาวางไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

X = load_data()

st.title("💳 Statlog German Credit Risk Prediction")
st.write("แอปพลิเคชันประเมินความเสี่ยงสินเชื่อ พร้อมระบบอธิบายโมเดลและความเป็นธรรม")

st.sidebar.header("กรอกข้อมูลผู้กู้")

# 4. สร้างฟอร์มรับค่าจากผู้ใช้สำหรับฟีเจอร์หลักๆ
input_data = {}
for col in X.columns[:10]:  # ดึงตัวอย่างฟีเจอร์มาทำฟอร์ม
    if X[col].dtype.name in ["category", "object"]:
        options = list(X[col].dropna().unique())
        input_data[col] = st.sidebar.selectbox(f"{col}", options)
    else:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        input_data[col] = st.sidebar.number_input(
            f"{col}", min_value=min_val, max_value=max_val, value=mean_val
        )

# เติมค่าที่เหลือด้วยค่าฐานนิยม/ค่าเฉลี่ย สำหรับคอลัมน์ที่ไม่ได้แสดงในฟอร์ม
for col in X.columns[10:]:
    if X[col].dtype.name in ["category", "object"]:
        input_data[col] = X[col].mode()[0]
    else:
        input_data[col] = X[col].mean()

# เรียงคอลัมน์ให้ตรงกับตอนเทรนโมเดล
input_df = pd.DataFrame([input_data])[X.columns]

# 5. ปุ่มทำนายผล
if st.button("🔍 ทำนายความเสี่ยงสินเชื่อ"):
    probas = model.predict_proba(input_df)[0]

    # หาตำแหน่งคลาส "ความเสี่ยงสูง (bad)" ให้ถูกต้อง ไม่ว่าโมเดลจะเข้ารหัสเป็น 'bad' หรือ 1
    classes = list(getattr(model, "classes_", []))
    bad_idx = 1
    for label in ("bad", "Bad", 1, "1"):
        if label in classes:
            bad_idx = classes.index(label)
            break
    proba = probas[bad_idx]

    # ใช้ Threshold ที่จูนไว้ (เช่น 0.5 หรือค่าที่ดีที่สุดจาก Colab)
    threshold = 0.5
    prediction = 1 if proba >= threshold else 0

    st.subheader("ผลการทำนาย:")
    if prediction == 1:
        st.error(f"⚠️ ความเสี่ยงสูง (Bad Risk) - โอกาสความเสี่ยง: {proba:.2f}")
    else:
        st.success(f"✅ ความเสี่ยงต่ำ (Good Risk) - โอกาสความเสี่ยง: {proba:.2f}")

    st.progress(float(proba))
    st.caption(f"Threshold ที่ใช้: {threshold}")

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(input_df.T.rename(columns={0: "ค่า"}))
