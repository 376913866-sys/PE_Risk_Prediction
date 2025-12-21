import streamlit as st
import pickle
import numpy as np
import os

# ---------- 模型路径（请保持不改，已与训练脚本统一） ----------
MODEL_PATH = os.path.expanduser("~/Desktop/PE_Risk_Prediction/model/rf_model.pkl")

# ---------- 加载模型 ----------
if not os.path.exists(MODEL_PATH):
    st.error("❌ 模型文件未找到：\n" + MODEL_PATH + "\n请先运行训练脚本生成 rf_model.pkl")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- 页面头部 ----------
st.title("🟣 子痫前期风险预测工具（Random Forest）")
st.markdown("请输入受试者的各项临床指标（全部 31 项），点击预测查看风险概率。")

# ---------- 特征列表 ----------
features_list = [
    "WBC", "N", "M", "Plt", "L", "LMR", "NMR", "SII", "PIV",
    "MoM值（P）", "MoM值（PI）", "MoM值（MAP）",
    "试管", "孕前BMI", "胎数", "产次",
    "AST", "ALT", "Cr", "UA", "HSI", "APRI", "FIB4", "SUA/sCr",
    "不良孕产史", "子痫前期既往史", "慢性高血压",
    "内科疾病史", "非典型抗磷脂综合征", "糖尿病", "妊娠年龄"
]

# ---------- 在侧边栏创建输入控件 ----------
st.sidebar.header("🔧 输入临床指标（全部 31 项）")
user_vals = []
for feat in features_list:
    if feat in ["试管", "不良孕产史", "子痫前期既往史", "慢性高血压", "内科疾病史", "非典型抗磷脂综合征", "糖尿病"]:
        val = st.sidebar.selectbox(feat + "（0或1）", [0, 1], index=0)
    else:
        # 合理默认值为0.0，可根据需要手动调整
        val = st.sidebar.number_input(feat, value=0.0, format="%.4f")
    user_vals.append(val)

# ---------- 预测 ----------
if st.sidebar.button("开始预测"):
    try:
        X = np.array(user_vals).reshape(1, -1)
        prob = float(model.predict_proba(X)[0, 1])
        percent = prob * 100
        st.subheader("预测结果")
        st.write(f"**子痫前期风险概率： {percent:.2f}%**")
        if prob >= 0.5:
            st.error("⚠️ 风险等级：高（≥ 0.5）")
        elif prob >= 0.2:
            st.warning("⚠ 风险等级：中（0.2 ≤ p < 0.5）")
        else:
            st.success("✅ 风险等级：低（< 0.2）")
    except Exception as e:
        st.error("预测时出错：\n" + str(e))
