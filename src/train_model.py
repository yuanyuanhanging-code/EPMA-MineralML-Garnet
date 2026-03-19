import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # 把 joblib 导入移到最上面
import os      # 导入 os 库来处理路径

# --- 1. 路径设置 (V2.0 核心修改) ---

# __file__ 是当前脚本 (train_model.py) 的完整路径
# os.path.dirname(__file__) 是 src 目录
# '..' 是返回上一级 (EPMA_ML 根目录)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 构造所有文件的绝对路径
# 注意：根据你的文件系统截图，文件名是 .CSV (大写)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'epma_master.CSV')
FIGURE_PATH = os.path.join(PROJECT_ROOT, 'figures', 'confusion_matrix.pdf')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rf_epma.pkl')
COLS_PATH = os.path.join(PROJECT_ROOT, 'models', 'req_cols.pkl')

# 自动创建尚不存在的目录
os.makedirs(os.path.join(PROJECT_ROOT, 'figures'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)

# -----------------------------------------------

# 1. 读数据 (使用新路径)
print(f"正在从 {DATA_PATH} 读取数据...")
df = pd.read_csv(DATA_PATH)

# 2. 剔除单样本类别
print("正在剔除单样本类别...")
df = df.groupby('Mineral').filter(lambda x: len(x) > 1)
print(f"数据清洗后剩余: {len(df)} 行")

# 3. 特征/标签
X = df.drop(columns=['PointID', 'Mineral'])
y = df['Mineral']

# 4. 拆分
print("正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. 模型
print("正在训练随机森林模型...")
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("模型训练完毕。")

# 6. 评估
y_pred = clf.predict(X_test)
print("\n--- 模型评估报告 ---")
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("---------------------\n")

# 7. 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()),
            linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('True', fontsize=14)
plt.title('Confusion Matrix (EPMA Mineral Classification)', fontsize=16)
plt.tight_layout()

# 8. 保存为矢量图 (使用新路径)
plt.savefig(FIGURE_PATH, format='pdf', bbox_inches='tight')
print(f"混淆矩阵已保存到: {FIGURE_PATH}")

# 9. 保存模型 (使用新路径)
joblib.dump(clf, MODEL_PATH)
print(f"模型快照已保存到: {MODEL_PATH}")

# 10. 保存训练时用的列顺序 (使用新路径)
joblib.dump(X_train.columns.tolist(), COLS_PATH)
print(f"模型列顺序已保存到: {COLS_PATH}")

# 在 Spyder 的 "Plots" 窗口中显示图像
plt.show()