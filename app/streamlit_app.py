import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd

# ================== 模型路径 ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")
LOG_MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")

# ================== 加载模型 ==================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(LOG_MODEL_PATH):
    st.error("❌ 模型文件未找到，请先在本地训练 RF 和 Logistic")
    st.stop()

rf_model = load_model(RF_MODEL_PATH)
log_model = load_model(LOG_MODEL_PATH)

# ================== 页面标题 ==================
st.title("🟣 子痫前期风险预测工具")
st.markdown("⚠️ **科研与教学用途，不用于临床诊断**")

# ================== 模型选择 ==================
model_choice = st.radio(
    "请选择预测模型：",
    ["随机森林（RF）", "Logistic 回归"],
    horizontal=True
)

# ================== 输入区 ==================
st.sidebar.header("🔧 输入临床指标")

WBC = st.sidebar.number_input("WBC", 0.0)
N = st.sidebar.number_input("中性粒细胞 N", 0.0)
L = st.sidebar.number_input("淋巴细胞 L", 0.0)
Plt = st.sidebar.number_input("血小板 Plt", 0.0)

AST = st.sidebar.number_input("AST", 0.0)
ALT = st.sidebar.number_input("ALT", 0.0)
UA = st.sidebar.number_input("尿酸 UA", 0.0)
Cr = st.sidebar.number_input("肌酐 Cr", 1.0)

age = st.sidebar.number_input("妊娠年龄", 30)

BMI = st.sidebar.number_input("孕前 BMI", 0.0)
IVF = st.sidebar.selectbox("试管", [0, 1])
chronic_htn = st.sidebar.selectbox("慢性高血压", [0, 1])
dm = st.sidebar.selectbox("糖尿病", [0, 1])
pe_history = st.sidebar.selectbox("子痫前期既往史", [0, 1])

# ================== 自动计算指标 ==================
LMR = L / WBC if WBC > 0 else 0
APRI = (AST / 40) * 100 / Plt if Plt > 0 else 0
FIB4 = (age * AST) / (Plt * np.sqrt(ALT)) if Plt > 0 and ALT > 0 else 0
HSI = 8 * ALT / AST + BMI if AST > 0 else 0
SUA_sCr = UA / Cr if Cr > 0 else 0

st.sidebar.markdown("### 📐 自动计算指标")
st.sidebar.write(f"LMR = {LMR:.3f}")
st.sidebar.write(f"APRI = {APRI:.3f}")
st.sidebar.write(f"FIB-4 = {FIB4:.3f}")
st.sidebar.write(f"HSI = {HSI:.3f}")
st.sidebar.write(f"SUA/sCr = {SUA_sCr:.3f}")

# ================== 特征向量（顺序必须与训练一致） ==================
features = np.array([[
    WBC, N, Plt, L,
    LMR,
    AST, ALT, UA, Cr,
    APRI, FIB4, HSI, SUA_sCr,
    BMI, IVF, chronic_htn, dm, pe_history, age
]])

# ================== 预测 ==================
if st.button("🚀 开始预测"):
    if model_choice == "随机森林（RF）":
        prob = rf_model.predict_proba(features)[0, 1]
        st.success(f"🌲 **随机森林预测风险：{prob*100:.1f}%**")

    else:
        prob = log_model.predict_proba(features)[0, 1]
        st.success(f"📈 **Logistic 回归预测风险：{prob*100:.1f}%**")

        # OR 解释
        coef = log_model.coef_[0]
        OR = np.exp(coef)

        st.subheader("📊 Logistic 回归 OR 解释（部分）")
        or_df = pd.DataFrame({
            "特征": [
                "WBC", "N", "Plt", "L", "LMR",
                "AST", "ALT", "UA", "Cr",
                "APRI", "FIB4", "HSI", "SUA/sCr",
                "BMI", "试管", "慢性高血压", "糖尿病", "既往PE", "年龄"
            ],
            "OR": OR
        })

        st.dataframe(or_df.round(3))
