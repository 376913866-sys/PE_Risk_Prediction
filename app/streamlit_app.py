import streamlit as st
import pickle
import numpy as np
import os

# ================== 页面基础设置 ==================
st.set_page_config(
    page_title="子痫前期风险预测工具",
    page_icon="🩺",
    layout="centered"
)

st.warning(
    """
⚠️ **免责声明**

本工具仅用于科研与教学演示目的，
预测结果不构成临床诊断或治疗建议。
请勿用于真实临床决策。
""",
    icon="⚠️"
)

# ================== 模型路径 ==================
MODEL_PATH = "model/rf_model.pkl"

# ================== 加载模型 ==================
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 模型文件未找到：\n{MODEL_PATH}\n请先运行 train_rf.py")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ================== 页面标题 ==================
st.title("🟣 子痫前期风险预测工具（Random Forest）")
st.markdown("请输入受试者的临床指标，点击预测查看子痫前期风险概率。")

# ================== 特征列表 ==================
features_list = [
    "WBC", "N", "M", "Plt", "L", "LMR", "NMR", "SII", "PIV",
    "MoM值（P）", "MoM值（PI）", "MoM值（MAP）",
    "试管", "孕前BMI", "胎数", "产次",
    "AST", "ALT", "Cr", "UA", "HSI", "APRI", "FIB4", "SUA/sCr",
    "不良孕产史", "子痫前期既往史", "慢性高血压",
    "内科疾病史", "非典型抗磷脂综合征", "糖尿病", "妊娠年龄"
]

binary_features = [
    "试管", "不良孕产史", "子痫前期既往史",
    "慢性高血压", "内科疾病史",
    "非典型抗磷脂综合征", "糖尿病"
]

# ================== 侧边栏输入 ==================
st.sidebar.header("🔧 输入临床指标")
user_vals = []

for feat in features_list:
    if feat in binary_features:
        val = st.sidebar.selectbox(f"{feat}（0=否, 1=是）", [0, 1], 
index=0)
    else:
        val = st.sidebar.number_input(feat, value=0.0)
    user_vals.append(val)

# ================== 🔥 预测逻辑（你问的就在这里） ==================
if st.sidebar.button("开始预测"):
    X = np.array(user_vals).reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])

    st.subheader("📊 预测结果")   # ✅ 就是加在这里
    st.metric(
        label="预测子痫前期风险概率",
        value=f"{prob*100:.2f}%"
    )

    if prob < 0.2:
        st.success("🟢 风险较低")
    elif prob < 0.5:
        st.warning("🟡 中等风险")
    else:
        st.error("🔴 高风险，请谨慎解读")

