import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# === 读取数据 ===
data_path = os.path.expanduser("~/Desktop/PE_Risk_Prediction/data/raw/20251130.csv")
df = pd.read_csv(data_path)

# === 目标变量 ===
target = "preeclampsia"   # 你的目标列

# === 特征列表 ===
features_list = [
    "WBC", "N", "M", "Plt", "L", "LMR", "NMR", "SII", "PIV",
    "MoM值（P）", "MoM值（PI）", "MoM值（MAP）",
    "试管", "孕前BMI", "胎数", "产次",
    "AST", "ALT", "Cr", "UA", "HSI", "APRI", "FIB4", "SUA/sCr",
    "不良孕产史", "子痫前期既往史", "慢性高血压",
    "内科疾病史", "非典型抗磷脂综合征",
    "糖尿病", "妊娠年龄"
]

# === 取特征和标签 ===
X = df[features_list].fillna(0)   # 自动填补缺失值
y = df[target]

# === 训练随机森林模型 ===
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

# === 保存模型 ===
model_path = os.path.expanduser("~/Desktop/PE_Risk_Prediction/model/rf_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("模型训练完成并保存至：", model_path)

