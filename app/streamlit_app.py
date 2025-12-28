import streamlit as st
import pickle
import numpy as np
import os
import math

# ===============================
# 模型路径
# ===============================
MODEL_PATH = "model/rf_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 模型文件未找到：{MODEL_PATH}")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===============================
# 免责声明
# ===============================
st.markdown(
    """
    <div style="background-color:#FFF8DC;padding:15px;border-radius:8px;">
    ⚠️ <b>免责声明</b><br>
    本工具仅用于科研与教学演示目的，预测结果不构成临床诊断或治疗建议。
    请勿用于真实临床决策。
    </div>
    """,
    unsafe_allow_html=True
)

st.title("🟣 子痫前期风险预测工具（Random Forest）")
st.markdown("请输入**基础原始指标**，系统将自动计算所有复合炎症/代谢指标。")

# ===============================
# 输入
# ===============================
st.sidebar.header("🔧 基础指标输入")

# —— 血液指标 ——
WBC = st.sidebar.number_input("WBC", value=0.0)
N   = st.sidebar.number_input("N（中性粒细胞）", value=0.0)
L   = st.sidebar.number_input("L（淋巴细胞）", value=0.0)
M   = st.sidebar.number_input("M（单核细胞）", value=0.0)
Plt = st.sidebar.number_input("Plt（血小板）", value=0.0)

# —— 生化 ——
AST = st.sidebar.number_input("AST", value=0.0)
ALT = st.sidebar.number_input("ALT", value=0.0)
UA  = st.sidebar.number_input("UA", value=0.0)
Cr  = st.sidebar.number_input("Cr", value=0.0)

# —— 人口学 ——
age = st.sidebar.number_input("年龄（Age）", value=30.0)
gest_age = st.sidebar.number_input("妊娠年龄（周）", value=0.0)
bmi = st.sidebar.number_input("孕前 BMI", value=0.0)

# —— MoM ——
mom_p   = st.sidebar.number_input("MoM值（P）", value=0.0)
mom_pi  = st.sidebar.number_input("MoM值（PI）", value=0.0)
mom_map = st.sidebar.number_input("MoM值（MAP）", value=0.0)

# —— 其他 ——
ivf = st.sidebar.selectbox("试管婴儿（0/1）", [0, 1])
fetus = st.sidebar.number_input("胎数", value=1.0)
parity = st.sidebar.number_input("产次", value=0.0)

bad_history = st.sidebar.selectbox("不良孕产史（0/1）", [0, 1])
pe_history = st.sidebar.selectbox("子痫前期既往史（0/1）", [0, 1])
chronic_htn = st.sidebar.selectbox("慢性高血压（0/1）", [0, 1])
internal_disease = st.sidebar.selectbox("内科疾病史（0/1）", [0, 1])
aps = st.sidebar.selectbox("非典型抗磷脂综合征（0/1）", [0, 1])
diabetes = st.sidebar.selectbox("糖尿病（0/1）", [0, 1])

# ===============================
# 自动计算指标
# ===============================
LMR = L / M if M > 0 else 0
NMR = N / M if M > 0 else 0
SII = (N * Plt / L) if L > 0 else 0
PIV = (N * Plt * M / L) if L > 0 else 0

APRI = ((AST / 40) / Plt * 100) if Plt > 0 else 0
FIB4 = (age * AST / (Plt * math.sqrt(ALT))) if (Plt > 0 and ALT > 0) else 0
HSI = (8 * ALT / AST + bmi) if AST > 0 else 0
SUA_sCr = UA / Cr if Cr > 0 else 0

with st.expander("📐 系统自动计算指标"):
    st.write(f"LMR = {LMR:.3f}")
    st.write(f"NMR = {NMR:.3f}")
    st.write(f"SII = {SII:.3f}")
    st.write(f"PIV = {PIV:.3f}")
    st.write(f"APRI = {APRI:.3f}")
    st.write(f"FIB-4 = {FIB4:.3f}")
    st.write(f"HSI = {HSI:.3f}")
    st.write(f"SUA/sCr = {SUA_sCr:.3f}")

# ===============================
# 预测
# ===============================
if st.sidebar.button("🚀 开始预测"):
    X = np.array([[
        WBC, N, M, Plt, L,
        LMR, NMR, SII, PIV,
        mom_p, mom_pi, mom_map,
        ivf, bmi, fetus, parity,
        AST, ALT, Cr, UA,
        HSI, APRI, FIB4, SUA_sCr,
        bad_history, pe_history, chronic_htn,
        internal_disease, aps, diabetes, gest_age
    ]])

    prob = model.predict_proba(X)[0, 1]

    st.subheader("📊 预测结果")
    st.metric("子痫前期风险概率", f"{prob*100:.2f}%")

    if prob >= 0.5:
        st.error("⚠️ 预测为高风险（仅科研参考）")
    else:
        st.success("✅ 预测为相对低风险（仅科研参考）")
