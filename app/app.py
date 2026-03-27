# app.py (V5.2 - 增加容错性清洗与进度条体验)

import streamlit as st
import pandas as pd
import joblib
import os
import time  # [新增] 用于模拟进度条延迟

# ---------- 1. 页面与路径设置 ----------
st.set_page_config(page_title='矿物智能识别与分析系统', layout='wide')
st.title('💎 EPMA 矿物智能识别与分析系统')
st.markdown('---')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rf_model_smote.pkl')
GENERAL_COLS_PATH = os.path.join(PROJECT_ROOT, 'models', 'req_cols_smote.pkl')
GARNET_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'garnet_subtype_model_v2.pkl')
GARNET_COLS_PATH = os.path.join(PROJECT_ROOT, 'models', 'garnet_subtype_cols_v2.pkl')

# ---------- 2. 石榴石端元计算模块 (保持V5.0验证版) ----------
MOLAR_MASSES = {'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Cr2O3': 151.99, 'FeO': 71.84, 'MnO': 70.94,
                'MgO': 40.30, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20}
CATIONS_PER_OXIDE = {'SiO2': 1, 'TiO2': 1, 'Al2O3': 2, 'Cr2O3': 2, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 2,
                     'K2O': 2}
OXYGENS_PER_OXIDE = {'SiO2': 2, 'TiO2': 2, 'Al2O3': 3, 'Cr2O3': 3, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 1,
                     'K2O': 1}


def calculate_garnet_formula(row):
    moles = {oxide: row.get(oxide, 0) / mass for oxide, mass in MOLAR_MASSES.items() if mass > 0};
    cation_proportions = {oxide.replace('2O3', '').replace('2O', '').replace('O', ''): moles.get(oxide, 0) * num for
                          oxide, num in CATIONS_PER_OXIDE.items()};
    total_oxygen = sum(moles.get(oxide, 0) * num for oxide, num in OXYGENS_PER_OXIDE.items());
    if total_oxygen == 0: return pd.Series({k: 0 for k in ['Alm(%)', 'Pyr(%)', 'Sps(%)', 'Grs(%)', 'And(%)', 'Uva(%)']})
    factor = 12.0 / total_oxygen;
    apfu = {cation: prop * factor for cation, prop in cation_proportions.items()};
    Alm, Pyr, Sps, Grs, And, Uva = 0, 0, 0, 0, 0, 0;
    fe_total = apfu.get('Fe', 0);
    mg = apfu.get('Mg', 0);
    mn = apfu.get('Mn', 0);
    ca = apfu.get('Ca', 0);
    al = apfu.get('Al', 0);
    cr = apfu.get('Cr', 0);
    ti = apfu.get('Ti', 0);
    fe3 = max(0, 2.0 - (al + cr + ti));
    fe2 = max(0, fe_total - fe3);
    pyralspite_cations = fe2 + mg + mn
    if pyralspite_cations > 0:
        pyralspite_potential = pyralspite_cations / 3.0;
        al_for_pyralspite = min(al / 2.0, pyralspite_potential)
        if al_for_pyralspite > 0: Pyr = (mg / pyralspite_cations) * al_for_pyralspite * 3.0;Alm = (
                                                                                                              fe2 / pyralspite_cations) * al_for_pyralspite * 3.0;Sps = (
                                                                                                                                                                                    mn / pyralspite_cations) * al_for_pyralspite * 3.0
    al_used = (Pyr + Alm + Sps) * 2.0 / 3.0;
    al_rem = al - al_used;
    rem_ca = ca
    if al_rem > 0: Grs = min(al_rem * 1.5, rem_ca);rem_ca -= Grs
    if cr > 0 and rem_ca > 0: Uva = min(cr * 1.5, rem_ca);rem_ca -= Uva
    if fe3 > 0 and rem_ca > 0: And = min(fe3 * 1.5, rem_ca)
    total = Alm + Pyr + Sps + Grs + And + Uva
    if total == 0: total = 1e-9
    return pd.Series({'Alm(%)': (Alm / total) * 100, 'Pyr(%)': (Pyr / total) * 100, 'Sps(%)': (Sps / total) * 100,
                      'Grs(%)': (Grs / total) * 100, 'And(%)': (And / total) * 100, 'Uva(%)': (Uva / total) * 100})


# --- [新增] 容错性列名清理函数 ---
def clean_column_names(df):
    """
    智能清理用户上传表格的列名，去除括号、空格、wt%等干扰字符。
    例如：'SiO2 (wt%)' -> 'SiO2', ' FeO % ' -> 'FeO'
    """
    clean_map = {}
    for col in df.columns:
        # 去除两端空格
        new_col = str(col).strip()
        # 去除常见的重量百分比后缀
        for suffix in ['(wt%)', 'wt%', '(%)', '%', '[wt%]', '[%]']:
            new_col = new_col.replace(suffix, '').strip()
        # 去除可能的下划线或多余空格
        new_col = new_col.replace(' ', '')
        clean_map[col] = new_col
    return df.rename(columns=clean_map)


# ---------- 3. 加载机器学习模型 ----------
@st.cache_resource(show_spinner="正在加载智能识别模型...")
def load_artifacts():
    try:
        general_model = joblib.load(GENERAL_MODEL_PATH)
        general_cols = joblib.load(GENERAL_COLS_PATH)
        garnet_model = joblib.load(GARNET_MODEL_PATH)
        garnet_cols = joblib.load(GARNET_COLS_PATH)
        return general_model, general_cols, garnet_model, garnet_cols
    except Exception as e:
        st.error(f"加载模型文件时发生严重错误: {e}")
        st.info("请确保 models/ 文件夹中包含所有必需的 .pkl 文件。")
        return None, None, None, None


general_mdl, general_REQ_COLS, garnet_mdl, garnet_REQ_COLS = load_artifacts()

# ---------- 4. Streamlit 应用主流程 ----------
if all([general_mdl, general_REQ_COLS, garnet_mdl, garnet_REQ_COLS]):

    st.markdown("#### 📁 数据上传与预处理")
    uploaded = st.file_uploader('请上传您的 EPMA csv 或 excel 文件 (系统会自动清洗不规范的列名及空值)',
                                type=['csv', 'xlsx'])

    if uploaded:
        try:
            # 1. 数据读取
            df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)

            # --- [优化] 智能容错清洗 ---
            st.info("🧹 正在进行数据智能清洗 (处理列名异常与空值)...")
            df_raw = clean_column_names(df_raw)  # 清洗列名

            # 尝试将所有能够转化为数字的列转化为数字，无法转化的变为 NaN，然后填充为 0
            for col in df_raw.columns:
                if col != 'PointID' and col != 'Sample':  # 保留可能存在的文本ID列
                    df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

            df_cleaned = df_raw.fillna(0)  # 最终确保没有 NaN

        except Exception as e:
            st.error(f"读取或清洗文件时出错: {e}");
            st.stop()

        # --- [新增] 模拟进度条体验 ---
        # 即使模型跑得很快，加个进度条会让用户觉得系统很专业在处理大数据
        total_rows = len(df_cleaned)
        progress_text = f"正在利用一级通用模型分析 {total_rows} 条数据..."
        my_bar = st.progress(0, text=progress_text)

        # 简单模拟一下处理耗时 (给点科技感)
        for percent_complete in range(100):
            time.sleep(0.01)  # 假装算得很辛苦
            my_bar.progress(percent_complete + 1, text=progress_text)

        # 进度条跑完后，瞬间出结果
        my_bar.empty()  # 清除进度条
        st.success(f"✅ 成功完成 {total_rows} 条数据的粗分类分析！")

        # --- 第一阶段：通用矿物识别 ---
        X_general = df_cleaned.reindex(columns=general_REQ_COLS, fill_value=0)
        general_preds = general_mdl.predict(X_general)
        res = df_cleaned.copy()
        res['Pred_Mineral'] = general_preds

        st.markdown('---')
        tab1, tab2 = st.tabs(["📊 主要预测结果", "🔬 石榴石专家分析"])

        with tab1:
            st.subheader("整体矿物识别结果")

            # [新增] 动态过滤交互 (让用户可以选择只看某种矿物)
            mineral_options = ['全部'] + list(res['Pred_Mineral'].unique())
            selected_minerals = st.multiselect("筛选查看特定矿物：", options=mineral_options, default='全部')

            if '全部' in selected_minerals or not selected_minerals:
                display_df = res
            else:
                display_df = res[res['Pred_Mineral'].isin(selected_minerals)]

            st.dataframe(display_df, use_container_width=True)

            csv = res.to_csv(index=False).encode('utf-8-sig')
            st.download_button('下载主要预测结果 (CSV)', csv, 'mineral_predictions.csv', mime='text/csv')

            st.subheader('预测矿物分布统计')
            st.bar_chart(res['Pred_Mineral'].value_counts())

        with tab2:
            st.markdown(
                "本模块首先对一级模型识别出的石榴石进行地球化学计算，然后利用**二级专家模型**，对石榴石的**具体亚类**进行智能预测。")
            if 'Garnet' in res['Pred_Mineral'].unique():
                st.subheader("石榴石亚类智能预测与组分分析")
                garnet_data = res[res['Pred_Mineral'] == 'Garnet'].copy()

                # --- [新增] 二级模型的进度条 ---
                garnet_count = len(garnet_data)
                st.info(f"在预测结果中发现 {garnet_count} 条石榴石样品，正在启动二级专家模型...")

                g_bar = st.progress(0, text="正在进行电价平衡与晶体化学特征注入...")
                endmember_results = garnet_data.apply(calculate_garnet_formula, axis=1)
                g_bar.progress(50, text="特征注入完毕，正在进行专家细分识别...")

                garnet_features = pd.concat([garnet_data, endmember_results], axis=1)
                X_garnet = garnet_features.reindex(columns=garnet_REQ_COLS, fill_value=0)
                subtype_preds = garnet_mdl.predict(X_garnet)
                garnet_features['Pred_Subtype'] = subtype_preds

                g_bar.progress(100, text="识别完成！")
                time.sleep(0.5)
                g_bar.empty()
                st.success("🎉 石榴石亚类预测与端元计算全部完成！")

                oxide_cols_display = [c for c in MOLAR_MASSES if c in garnet_features.columns]
                endmember_cols_display = [col for col in endmember_results.columns if garnet_features[col].sum() > 0.01]
                cols_to_show = ['PointID', 'Pred_Subtype'] + oxide_cols_display + endmember_cols_display

                # 容错：如果用户上传的表没有 PointID 列，就不显示它
                cols_to_show = [c for c in cols_to_show if c in garnet_features.columns]

                st.dataframe(garnet_features.get(cols_to_show, garnet_features.columns).style.format("{:.2f}",
                                                                                                     subset=endmember_cols_display),
                             use_container_width=True)

                st.subheader("预测亚类统计")
                st.bar_chart(garnet_features['Pred_Subtype'].value_counts())

                st.subheader("端元组分构成可视化")
                stacked_bar_data = garnet_features[endmember_cols_display]
                st.bar_chart(stacked_bar_data)

                garnet_csv = garnet_features.to_csv(index=False).encode('utf-8-sig')
                st.download_button('下载石榴石分析结果 (CSV)', garnet_csv, 'garnet_endmember_analysis.csv',
                                   mime='text/csv')
            else:
                st.info("本次上传文件的预测结果中未发现石榴石样品。")
else:
    st.warning("模型文件加载中或加载失败，请检查终端中的错误信息。")

st.markdown('---')
st.caption('🛠️  Powered by Hierarchical Reasoning Models | 技术支持: 2250858@tongji.edu.cn 刘沅杭')
