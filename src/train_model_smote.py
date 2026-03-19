# train_model_smote.py (修正版 V2 - 可直接运行)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
import os
import numpy as np  # 修正：导入 numpy 库

# --- 1. 路径设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'epma_master.CSV')
FIGURE_PATH_SMOTE = os.path.join(PROJECT_ROOT, 'figures', 'cm_smote.pdf')
MODEL_PATH_SMOTE = os.path.join(PROJECT_ROOT, 'models', 'rf_model_smote.pkl')
COLS_PATH_SMOTE = os.path.join(PROJECT_ROOT, 'models', 'req_cols_smote.pkl')

os.makedirs(os.path.join(PROJECT_ROOT, 'figures'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

# --- 2. 数据加载与初步清洗 ---
print(f"正在从 {DATA_PATH} 读取数据...")
df = pd.read_csv(DATA_PATH)

# 先剔除在整个数据集中都无法进行分层抽样的类别
df = df.groupby('Mineral').filter(lambda x: len(x) > 1)
print(f"剔除单一样本类别后，剩余数据: {len(df)} 行")

X = df.drop(columns=['PointID', 'Mineral'])
y = df['Mineral']

# --- 3. 划分训练集和测试集 (提前进行) ---
print("正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. SMOTE 过采样 ---
print("正在对训练集进行 SMOTE 过采样...")

# 在训练集内部处理样本过少的问题
k_neighbors_max = 5 # 我们期望的最大邻居数
counts_train = y_train.value_counts()
under_represented_classes = counts_train[counts_train <= k_neighbors_max].index

if not under_represented_classes.empty:
    print(f"发现训练集中样本过少的类别: {list(under_represented_classes)}，正在通过复制来补充...")
    for cls in under_represented_classes:
        num_samples = counts_train[cls]
        num_to_add = k_neighbors_max + 1 - num_samples
        
        class_indices = y_train[y_train == cls].index
        indices_to_add = np.random.choice(class_indices, size=num_to_add, replace=True)
        
        X_train = pd.concat([X_train, X_train.loc[indices_to_add]], ignore_index=True)
        y_train = pd.concat([y_train, y_train.loc[indices_to_add]], ignore_index=True)

# 动态设定SMOTE的k_neighbors参数
min_class_count_in_train = y_train.value_counts().min()
k_neighbors_for_smote = min(k_neighbors_max, min_class_count_in_train - 1)

print(f"训练集中最小类别样本数已扩充至: {min_class_count_in_train}，SMOTE邻居数 k_neighbors 设为: {k_neighbors_for_smote}")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors_for_smote)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"SMOTE 处理后训练集大小: {X_train_sm.shape[0]} 行")

# --- 5. 训练、评估、画图 ---
print("正在训练 SMOTE 随机森林模型...")
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train_sm, y_train_sm)
print("模型训练完毕。")

y_pred = clf.predict(X_test)
print("\n--- SMOTE 模型评估报告 (在原始测试集上) ---")
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print("-------------------------------------------\n")

all_classes = sorted(list(set(y_test.unique()).union(set(y_pred))))
cm = confusion_matrix(y_test, y_pred, labels=all_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=all_classes, yticklabels=all_classes,
            linewidths=0.5, square=True, cbar_kws={'shrink': .8})
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix (SMOTE-balanced Model)')
plt.tight_layout()

# --- 6. 保存结果 ---
plt.savefig(FIGURE_PATH_SMOTE, bbox_inches='tight')
print(f"SMOTE 混淆矩阵已保存到: {FIGURE_PATH_SMOTE}")

joblib.dump(clf, MODEL_PATH_SMOTE)
print(f"SMOTE 模型快照已保存到: {MODEL_PATH_SMOTE}")

joblib.dump(X_train.columns.tolist(), COLS_PATH_SMOTE)
print(f"模型列顺序已保存到: {COLS_PATH_SMOTE}")

plt.show()
