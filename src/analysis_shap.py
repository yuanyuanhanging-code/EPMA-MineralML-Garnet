# src/analysis_shap_interactive.py
# 功能：交互式生成石榴石亚类 SHAP 蜂群图
# 亮点：运行一次，无限出图，支持手动输入选择

import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 1. 配置路径与字体 ---
# 自动定位项目根目录 (假设脚本在 src/ 下)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'garnet_subtype_model_v2.pkl')
DATA_PATH_X = os.path.join(PROJECT_ROOT, 'models', 'garnet_X_train.pkl')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# 设置中文字体 (Windows: SimHei, Mac: Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 加载资源 ---
print("=" * 60)
print("💎 石榴石专家模型 - SHAP 可解释性分析工具 (交互版)")
print("=" * 60)
print(f"1. 正在加载模型与数据...")

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH_X):
    print("❌ 错误：找不到模型或数据快照！")
    print(f"请检查 models 文件夹下是否有: {os.path.basename(MODEL_PATH)} 和 {os.path.basename(DATA_PATH_X)}")
    sys.exit()

clf = joblib.load(MODEL_PATH)
X_data = joblib.load(DATA_PATH_X)
print(f"✅ 模型加载成功！训练集样本数: {len(X_data)}")

# --- 3. 计算 SHAP 值 (最耗时的一步，只做一次) ---
SAMPLE_SIZE = 500  # 采样数量，太大会慢
print(f"2. 正在计算全局 SHAP 值 (采样 {SAMPLE_SIZE} 个样本)...")
print("   (这可能需要 1-2 分钟，请耐心等待...)")

X_sample = X_data.sample(n=min(SAMPLE_SIZE, len(X_data)), random_state=42)
explainer = shap.TreeExplainer(clf)
# 关闭可加性检查以避免报错
shap_values_raw = explainer.shap_values(X_sample, check_additivity=False)
print("✅ SHAP 计算完成！")

# 获取所有类别名称
class_names = list(clf.classes_)

# --- 4. 交互式绘图循环 ---
while True:
    print("-" * 60)
    print(f"【模型包含的亚类列表 ({len(class_names)}个)】:")
    
    # 打印菜单
    for i, name in enumerate(class_names):
        # 每行打印 3 个，排版好看点
        end_char = '\n' if (i + 1) % 3 == 0 else '\t\t'
        print(f"[{i}] {name}", end=end_char)
    print("\n" + "-" * 60)
    
    # 获取用户输入
    user_input = input("👉 请输入【序号】(例如 0) 或【名称】(例如 镁铝榴石)，输入 'q' 退出: ").strip()
    
    if user_input.lower() == 'q':
        print("👋 程序已退出。")
        break
    
    # 解析输入
    target_index = -1
    target_name = ""
    
    try:
        # 尝试解析为数字索引
        idx = int(user_input)
        if 0 <= idx < len(class_names):
            target_index = idx
            target_name = class_names[idx]
        else:
            print(f"❌ 索引越界！请输入 0 到 {len(class_names)-1} 之间的数字。")
            continue
    except ValueError:
        # 尝试解析为名称 (支持模糊搜索)
        found = False
        for i, name in enumerate(class_names):
            if user_input in name: # 只要包含关键词就算找到
                target_index = i
                target_name = name
                found = True
                break
        if not found:
            print(f"❌ 未找到包含 '{user_input}' 的类别，请重试。")
            continue

    print(f"\n🚀 正在生成 [{target_name}] 的 SHAP 蜂群图...")
    
    # --- 提取对应类别的 SHAP 矩阵 (维度适配) ---
    shap_matrix = None
    if isinstance(shap_values_raw, list):
        shap_matrix = shap_values_raw[target_index]
    elif isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
        shap_matrix = shap_values_raw[:, :, target_index]
    else:
        shap_matrix = shap_values_raw # 二分类情况
        
    # 形状防御性检查
    if shap_matrix.shape != X_sample.shape:
        if shap_matrix.T.shape == X_sample.shape:
            shap_matrix = shap_matrix.T
            
    # --- 绘图 ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_matrix, X_sample, show=False)
    
    # 设置标题
    plt.title(f"SHAP Summary: {target_name}", fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    # 文件名处理：把不能做文件名的字符去掉
    safe_name = target_name.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(FIGURE_DIR, f'shap_summary_{safe_name}.png')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"💾 图片已保存至: {save_path}")
    
    # 展示图片 (可能会阻塞循环，关闭图片窗口后继续)
    plt.show() 
    print("✅ 绘图完成！(如未看到图片，请到 figures 文件夹查看)")