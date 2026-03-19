# src/train_garnet_subtype_model.py (V6.0 - 终极专业版)
# 功能：
# 1. 动态加载多源数据
# 2. 内置晶体化学计算引擎 (Grew 2013)
# 3. 生成符合中文地质规范的命名标签 (如：富铁镁铝榴石)
# 4. 执行 SMOTE 数据平衡
# 5. 训练随机森林模型并评估
# 6. 保存模型及训练数据快照 (供 SHAP 分析使用)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')

# --- 1. 基础配置 ---
# 设置绘图字体 (Windows: SimHei, Mac: Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')

# 确保文件夹存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'garnet_subtype_model_v2.pkl')
COLS_PATH = os.path.join(MODEL_DIR, 'garnet_subtype_cols_v2.pkl')
# SHAP 分析专用数据路径
SHAP_X_PATH = os.path.join(MODEL_DIR, 'garnet_X_train.pkl')
SHAP_Y_PATH = os.path.join(MODEL_DIR, 'garnet_y_train.pkl')

# --- 2. 晶体化学计算模块 (Grew 2013 规则) ---
MOLAR_MASSES = {'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Cr2O3': 151.99, 'FeO': 71.84, 'MnO': 70.94,
                'MgO': 40.30, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20}
CATIONS_PER_OXIDE = {'SiO2': 1, 'TiO2': 1, 'Al2O3': 2, 'Cr2O3': 2, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 2,
                     'K2O': 2}
OXYGENS_PER_OXIDE = {'SiO2': 2, 'TiO2': 2, 'Al2O3': 3, 'Cr2O3': 3, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 1,
                     'K2O': 1}


def calculate_garnet_formula(row):
    """
    基于 12 氧原子法计算石榴石端元 (Grew et al., 2013)
    """
    moles = {oxide: row.get(oxide, 0) / mass for oxide, mass in MOLAR_MASSES.items() if mass > 0}
    # 简单的总量检查
    total_oxygen = sum(moles.get(oxide, 0) * num for oxide, num in OXYGENS_PER_OXIDE.items())

    # 防止除零错误
    if total_oxygen == 0:
        return pd.Series({k: 0 for k in ['Alm(%)', 'Pyr(%)', 'Sps(%)', 'Grs(%)', 'And(%)', 'Uva(%)']})

    factor = 12.0 / total_oxygen
    # 计算阳离子数 (apfu)
    cation_proportions = {oxide.replace('2O3', '').replace('2O', '').replace('O', ''): moles.get(oxide, 0) * num for
                          oxide, num in CATIONS_PER_OXIDE.items()}
    apfu = {cation: prop * factor for cation, prop in cation_proportions.items()}

    Alm, Pyr, Sps, Grs, And, Uva = 0, 0, 0, 0, 0, 0

    # 提取主要阳离子
    fe_total = apfu.get('Fe', 0);
    mg = apfu.get('Mg', 0);
    mn = apfu.get('Mn', 0);
    ca = apfu.get('Ca', 0)
    al = apfu.get('Al', 0);
    cr = apfu.get('Cr', 0);
    ti = apfu.get('Ti', 0)

    # --- 核心逻辑：电价平衡计算 Fe3+ ---
    # 假设 Y 位 (八面体) 由 Al, Cr, Ti, Fe3+ 填充，理想和为 2.0
    fe3 = max(0, 2.0 - (al + cr + ti))
    # 剩余的铁为 Fe2+
    fe2 = max(0, fe_total - fe3)

    # 1. 铝榴石系列 (Pyralspite): X位=Mg,Fe2+,Mn; Y位=Al
    pyralspite_cations = fe2 + mg + mn
    if pyralspite_cations > 0:
        # 分配给 Pyralspite 的 Al
        pyralspite_potential = pyralspite_cations / 3.0
        al_for_pyralspite = min(al / 2.0, pyralspite_potential)

        if al_for_pyralspite > 0:
            Pyr = (mg / pyralspite_cations) * al_for_pyralspite * 3.0
            Alm = (fe2 / pyralspite_cations) * al_for_pyralspite * 3.0
            Sps = (mn / pyralspite_cations) * al_for_pyralspite * 3.0

    # 2. 钙榴石系列 (Ugrandite): X位=Ca; Y位=Al,Fe3+,Cr
    al_used = (Pyr + Alm + Sps) * 2.0 / 3.0
    al_rem = al - al_used
    rem_ca = ca

    # 钙铝榴石 (Grossular): Ca3 Al2
    if al_rem > 0:
        Grs = min(al_rem * 1.5, rem_ca)
        rem_ca -= Grs
    # 钙铬榴石 (Uvarovite): Ca3 Cr2
    if cr > 0 and rem_ca > 0:
        Uva = min(cr * 1.5, rem_ca)
        rem_ca -= Uva
    # 钙铁榴石 (Andradite): Ca3 Fe3+2
    if fe3 > 0 and rem_ca > 0:
        And = min(fe3 * 1.5, rem_ca)

    # 归一化
    total = Alm + Pyr + Sps + Grs + And + Uva
    if total == 0: total = 1e-9

    return pd.Series({
        'Alm(%)': (Alm / total) * 100, 'Pyr(%)': (Pyr / total) * 100, 'Sps(%)': (Sps / total) * 100,
        'Grs(%)': (Grs / total) * 100, 'And(%)': (And / total) * 100, 'Uva(%)': (Uva / total) * 100
    })


# --- 3. [专业版] 标签命名函数 ---
def generate_subtype_name(row):
    """
    生成符合地质学学术规范的亚类名称。
    格式：富[修饰成分][主要矿物名称]
    规则：当次要端元含量 >= 10% 时添加修饰词。
    """
    # 1. 定义主要矿物名称 (名词)
    primary_names = {
        'Alm(%)': '铁铝榴石', 'Pyr(%)': '镁铝榴石', 'Sps(%)': '锰铝榴石',
        'Grs(%)': '钙铝榴石', 'And(%)': '钙铁榴石', 'Uva(%)': '钙铬榴石'
    }

    # 2. 定义修饰词 (形容词)
    modifiers = {
        'Alm(%)': '富铁', 'Pyr(%)': '富镁', 'Sps(%)': '富锰',
        'Grs(%)': '富钙', 'And(%)': '富铁', 'Uva(%)': '富铬'
    }

    # 排序：找出 No.1 和 No.2
    sorted_endmembers = sorted(primary_names.keys(), key=lambda k: row[k], reverse=True)
    primary_key = sorted_endmembers[0]
    secondary_key = sorted_endmembers[1]

    primary_name = primary_names[primary_key]

    # 阈值判断 (10%)
    if row[secondary_key] < 10:
        return primary_name  # 纯端元，如“镁铝榴石”
    else:
        # 默认修饰词
        modifier = modifiers[secondary_key]

        # 特殊逻辑修正：钙系内部互为次要时，需区分 Fe3+ 和 Al
        # 如果是 钙铝(Grs) 和 钙铁(And) 混合
        if primary_key == 'Grs(%)' and secondary_key == 'And(%)':
            modifier = '富铁'  # 强调 Fe3+ (Andradite组分)
        elif primary_key == 'And(%)' and secondary_key == 'Grs(%)':
            modifier = '富铝'  # 强调 Al (Grossular组分)

        return f"{modifier}{primary_name}"


# --- 主程序 ---
if __name__ == "__main__":
    print("=" * 60)
    print("💎 二级专家模型训练系统 (V6.0 专业命名版)")
    print("=" * 60)

    # 1. 动态加载并合并数据
    print(f"[1/7] 正在加载数据...")
    df_master = pd.read_csv(os.path.join(DATA_DIR, 'epma_master.CSV'))
    # 只取石榴石数据
    df_garnet_master = df_master[df_master['Mineral'] == 'Garnet'].copy()

    # 扫描其他清洗过的数据集
    cleaned_files = glob.glob(os.path.join(DATA_DIR, '*_cleaned.csv'))
    all_garnet_dfs = [df_garnet_master]

    for file_path in cleaned_files:
        try:
            df_new = pd.read_csv(file_path)
            all_garnet_dfs.append(df_new)
            print(f"   - 已合并: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"   - 错误: 无法读取 {file_path}, {e}")

    df_combined = pd.concat(all_garnet_dfs, ignore_index=True, sort=False).fillna(0)
    print(f"   >>> 原始数据总量: {len(df_combined)} 条")

    # 2. 特征工程 (计算端元 + 生成新标签)
    print(f"\n[2/7] 执行特征工程与标签生成...")
    # 应用晶体化学计算
    endmember_results = df_combined.apply(calculate_garnet_formula, axis=1)
    # 生成专业标签
    df_combined['Subtype_Name'] = endmember_results.apply(generate_subtype_name, axis=1)
    # 合并特征
    df_processed = pd.concat([df_combined, endmember_results], axis=1)

    print("\n   --- 标签分布概览 (Top 10) ---")
    print(df_processed['Subtype_Name'].value_counts().head(10))

    # 3. 定义特征与过滤
    oxide_cols = list(MOLAR_MASSES.keys())
    endmember_cols = ['Alm(%)', 'Pyr(%)', 'Sps(%)', 'Grs(%)', 'And(%)', 'Uva(%)']
    feature_cols = [col for col in oxide_cols if col in df_processed.columns] + endmember_cols

    print(f"\n[3/7] 样本清洗与过滤...")
    class_counts = df_processed['Subtype_Name'].value_counts()

    # 设定阈值为 6 (确保 SMOTE k=5 可运行，且能分层抽样)
    MIN_SAMPLES_THRESHOLD = 6
    valid_classes = class_counts[class_counts >= MIN_SAMPLES_THRESHOLD].index

    df_final = df_processed[df_processed['Subtype_Name'].isin(valid_classes)].copy()
    dropped_classes = class_counts[class_counts < MIN_SAMPLES_THRESHOLD]

    if not dropped_classes.empty:
        print(f"   ⚠️ 已剔除 {len(dropped_classes)} 个样本极少(<{MIN_SAMPLES_THRESHOLD})的孤儿类别")
        # print(dropped_classes.index.tolist()) # 如果想看具体剔除了谁，取消注释

    X = df_final[feature_cols]
    y = df_final['Subtype_Name']

    # 4. SMOTE 增强
    print(f"\n[4/7] 执行 SMOTE 数据平衡...")
    # 动态设定 k
    min_samples = y.value_counts().min()
    k_neighbors = min(5, min_samples - 1)

    try:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"   ✅ SMOTE 成功! 样本量: {len(X)} -> {len(X_resampled)}")
    except Exception as e:
        print(f"   ❌ SMOTE 失败 (可能类别太少), 使用原始数据: {e}")
        X_resampled, y_resampled = X, y

    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # 6. 训练模型
    print(f"\n[5/7] 训练随机森林分类器...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    print("   ✅ 训练完毕。")

    # 7. 评估报告
    print(f"\n[6/7] 模型评估报告 (测试集)...")
    print("-" * 60)
    # 这里直接打印，中文标签应该能正常显示
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))
    print("-" * 60)

    # 8. 特征重要性可视化
    print(f"\n[7/7] 生成图表与保存模型...")
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': clf.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Top 20 Features (Garnet Expert Model)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'garnet_feature_importance.pdf'))

    # 保存模型
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(feature_cols, COLS_PATH)

    # 【与SHAP联动】保存训练数据快照
    joblib.dump(X_train, SHAP_X_PATH)
    joblib.dump(y_train, SHAP_Y_PATH)

    print(f"   ✅ 模型已保存至: {MODEL_PATH}")
    print(f"   ✅ SHAP数据已保存至: {SHAP_X_PATH}")
    print("\n🎉 全部流程执行成功！您可以运行 SHAP 分析脚本了。")