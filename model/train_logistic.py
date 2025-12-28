import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------- 1. 读取数据 ----------
df = pd.read_csv("../data/raw/20251130.csv")

# ---------- 2. 目标变量 ----------
y = df["preeclampsia"]

# ---------- 3. 特征 ----------
X = df.drop(columns=["preeclampsia"])

# ① 删除字符串列
X = X.select_dtypes(exclude=["object"])

# ② 用中位数填补 NaN（🔥关键）
X = X.fillna(X.median())

print("✅ Logistic 使用的特征数：", X.shape[1])
print("是否还有 NaN：", X.isna().any().any())

# ---------- 4. 划分 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 5. 模型 ----------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# ---------- 6. 保存 ----------
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 Logistic 回归模型训练完成并保存")



