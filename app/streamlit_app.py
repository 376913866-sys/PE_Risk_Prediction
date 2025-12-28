import streamlit as st
import pickle
import numpy as np
import os
import pandas as pd
import math

# ================== 模型路径 ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_MODEL_PATH = os.path.join(BASE_DIR, "model", "rf_model.pkl")
LOG_MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")

# ================== 加载模型 ==================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

rf_model = load_model(RF_MODEL_PATH)
log_model = load_model(LOG_MODEL_PATH)

# 获取模型训练特征
rf_feature_cols = rf_model.feature_names_in_
log_feature_cols = log_model.feature_names_in_

# ================== 页面标题 ==================
st.title("🟣 子痫前期风险预测工具")
st.markdown("⚠️ **科研与教学用途，不用于临床诊断**")

# ================== 模型选择 ==================
model_choice = st.radio(
    "请选择预测模型：",
    ["随机森林（RF）", "Logistic 回归"],
    horizontal=True
)

# ================== 预测方式选择 ==================
mode = st.radio(
    "请选择预测方式：",
    ["单条输入预测", "CSV 批量预测"],
    horizontal=True
)

# ================== 批量预测风险等级函数 ==================
def get_risk_level(prob):
    if prob < 0.2:
        return "低风险"
    elif prob < 0.5:
        return "中风险"
    else:
        return "高风险"

# ================== 获取当前模型特征列 ==================
def get_model_feature_cols():
    return rf_feature_cols if model_choice == "随机森林（RF）" else log_feature_cols

# ================== 单条输入预测 ==================
if mode == "单条输入预测":
    st.sidebar.header("🔧 输入临床指标")
    WBC = st.sidebar.number_input("WBC", 0.0)
    N = st.sidebar.number_input("中性粒细胞 N", 0.0)
    L = st.sidebar.number_input("淋巴细胞 L", 0.0)
    M = st.sidebar.number_input("单核细胞 M", 0.0)
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

    # 自动计算衍生指标
    LMR = L / M if M > 0 else 0
    NMR = N / M if M > 0 else 0
    SII = (N * Plt / L) if L > 0 else 0
    PIV = (N * Plt * M / L) if L > 0 else 0
    APRI = ((AST / 40) / Plt * 100) if Plt > 0 else 0
    FIB4 = (age * AST / (Plt * math.sqrt(ALT))) if (Plt > 0 and ALT > 0) else 0
    HSI = (8 * ALT / AST + BMI) if AST > 0 else 0
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

    # 构造输入字典
    input_dict = {
        "WBC": WBC, "N": N, "Plt": Plt, "L": L, "M": M,
        "LMR": LMR, "NMR": NMR, "SII": SII, "PIV": PIV,
        "AST": AST, "ALT": ALT, "UA": UA, "Cr": Cr,
        "APRI": APRI, "FIB4": FIB4, "HSI": HSI, "SUA/sCr": SUA_sCr,
        "BMI": BMI, "孕前BMI": BMI, "试管": IVF, "慢性高血压": chronic_htn,
        "糖尿病": dm, "子痫前期既往史": pe_history, "妊娠年龄": age
        # 其余特征填 0
    }

    feature_cols = get_model_feature_cols()
    features = np.array([input_dict.get(feat, 0) for feat in feature_cols]).reshape(1, -1)

    if st.button("🚀 开始预测"):
        try:
            if model_choice == "随机森林（RF）":
                prob = rf_model.predict_proba(features)[0, 1]
            else:
                prob = log_model.predict_proba(features)[0, 1]
            st.success(f"预测风险概率：{prob*100:.1f}% ({get_risk_level(prob)})")
        except ValueError as e:
            st.error(f"❌ 预测失败: {e}")

# ================== CSV 批量预测 ==================
else:
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # 自动计算衍生指标
        df["LMR"] = df["L"] / df["M"].replace(0, np.nan)
        df["NMR"] = df["N"] / df["M"].replace(0, np.nan)
        df["SII"] = (df["N"] * df["Plt"] / df["L"]).replace(np.inf, 0).fillna(0)
        df["PIV"] = (df["N"] * df["Plt"] * df["M"] / df["L"]).replace(np.inf, 0).fillna(0)
        df["APRI"] = ((df["AST"] / 40) / df["Plt"] * 100).replace(np.inf, 0).fillna(0)
        df["FIB4"] = (df["妊娠年龄"] * df["AST"] / (df["Plt"] * np.sqrt(df["ALT"]))).replace(np.inf, 0).fillna(0)
        df["HSI"] = (8 * df["ALT"] / df["AST"] + df["孕前 BMI"]).replace(np.inf, 0).fillna(0)
        df["SUA_sCr"] = (df["UA"] / df["Cr"]).replace(np.inf, 0).fillna(0)

        # 获取当前模型特征
        feature_cols = get_model_feature_cols()

        # 检测缺失列
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.warning("⚠️ 以下特征列缺失，将使用 0 填充：")
            for col in missing_cols:
                st.write("-", col)
                df[col] = 0

        # 构造特征矩阵
        X = df[feature_cols].values

        try:
            if model_choice == "随机森林（RF）":
                probs = rf_model.predict_proba(X)[:, 1]
            else:
                probs = log_model.predict_proba(X)[:, 1]

            df["预测风险概率"] = probs
            df["风险等级"] = [get_risk_level(p) for p in probs]

            st.success("✅ 批量预测完成！")
            st.dataframe(df)
            st.download_button("📥 下载结果 CSV", df.to_csv(index=False).encode('utf-8'), "预测结果.csv")
        except ValueError as e:
            st.error(f"❌ 预测失败: {e}")
