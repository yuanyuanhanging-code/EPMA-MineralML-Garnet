# app.py (V5.7 - MLOps 终极版：加入模型持续进化与版本控制机制)

import streamlit as st
import pandas as pd
import joblib
import os
import time
import sqlite3
import json
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# ---------- 1. 页面与路径设置 ----------
st.set_page_config(page_title='矿物智能识别与分析系统', layout='wide', page_icon='💎')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'epma_master.CSV')  # 原始主数据集
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'epma_history.db')

# 模型版本控制路径
GENERAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rf_model_smote.pkl')  # 当前工作模型
FACTORY_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rf_model_smote_factory.pkl')  # 出厂模型
PREV_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rf_model_smote_prev.pkl')  # 上一版本备份
GENERAL_COLS_PATH = os.path.join(PROJECT_ROOT, 'models', 'req_cols_smote.pkl')

GARNET_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'garnet_subtype_model_v2.pkl')
GARNET_COLS_PATH = os.path.join(PROJECT_ROOT, 'models', 'garnet_subtype_cols_v2.pkl')

# ---------- 2. 状态与环境初始化 ----------
if 'processed_file' not in st.session_state: st.session_state.processed_file = None
if 'general_res' not in st.session_state: st.session_state.general_res = None
if 'garnet_res' not in st.session_state: st.session_state.garnet_res = None

# 初始化出厂模型备份 (如果不存在，自动把当前模型备份为出厂模型)
if not os.path.exists(FACTORY_MODEL_PATH) and os.path.exists(GENERAL_MODEL_PATH):
    shutil.copy(GENERAL_MODEL_PATH, FACTORY_MODEL_PATH)


# ---------- 3. 数据库管理 (新增增量训练表) ----------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 历史预测记录表
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     timestamp
                     TEXT,
                     filename
                     TEXT,
                     data_json
                     TEXT
                 )''')
    # 用户上传的增量训练集表
    c.execute('''CREATE TABLE IF NOT EXISTS user_training
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     batch_id
                     TEXT,
                     timestamp
                     TEXT,
                     data_json
                     TEXT
                 )''')
    conn.commit()
    conn.close()


# 基础历史记录功能
def save_history(filename, df_general, df_garnet):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    combined_data = {"general": df_general.to_dict(orient='records'),
                     "garnet": df_garnet.to_dict(orient='records') if df_garnet is not None else None}
    c.execute("INSERT INTO history (timestamp, filename, data_json) VALUES (?, ?, ?)",
              (timestamp, filename, json.dumps(combined_data)))
    conn.commit()
    conn.close()


def get_history_list():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, timestamp, filename FROM history ORDER BY id DESC", conn)
    conn.close()
    return df


def clear_all_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history");
    c.execute("DELETE FROM sqlite_sequence WHERE name='history'")
    conn.commit()
    conn.close()


# --- MLOps 增量训练功能库 ---
def save_user_training_data(df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    batch_id = datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO user_training (batch_id, timestamp, data_json) VALUES (?, ?, ?)",
              (batch_id, timestamp, df.to_json(orient='records')))
    conn.commit()
    conn.close()
    return batch_id


def get_all_user_training_data():
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql_query("SELECT data_json FROM user_training", conn)
    conn.close()
    if df_meta.empty: return pd.DataFrame()

    dfs = [pd.read_json(row['data_json'], orient='records') for _, row in df_meta.iterrows()]
    return pd.concat(dfs, ignore_index=True)


def delete_last_training_batch():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM user_training WHERE id = (SELECT MAX(id) FROM user_training)")
    conn.commit()
    conn.close()


def clear_user_training_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM user_training");
    c.execute("DELETE FROM sqlite_sequence WHERE name='user_training'")
    conn.commit()
    conn.close()


init_db()

# ---------- 4. 核心渲染引擎与清洗模块 ----------
MOLAR_MASSES = {'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Cr2O3': 151.99, 'FeO': 71.84, 'MnO': 70.94,
                'MgO': 40.30, 'CaO': 56.08, 'Na2O': 61.98, 'K2O': 94.20}
CATIONS_PER_OXIDE = {'SiO2': 1, 'TiO2': 1, 'Al2O3': 2, 'Cr2O3': 2, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 2,
                     'K2O': 2}
OXYGENS_PER_OXIDE = {'SiO2': 2, 'TiO2': 2, 'Al2O3': 3, 'Cr2O3': 3, 'FeO': 1, 'MnO': 1, 'MgO': 1, 'CaO': 1, 'Na2O': 1,
                     'K2O': 1}


def calculate_garnet_formula(row):
    moles = {oxide: row.get(oxide, 0) / mass for oxide, mass in MOLAR_MASSES.items() if mass > 0}
    cation_proportions = {oxide.replace('2O3', '').replace('2O', '').replace('O', ''): moles.get(oxide, 0) * num for
                          oxide, num in CATIONS_PER_OXIDE.items()}
    total_oxygen = sum(moles.get(oxide, 0) * num for oxide, num in OXYGENS_PER_OXIDE.items())
    if total_oxygen == 0: return pd.Series({k: 0 for k in ['Alm(%)', 'Pyr(%)', 'Sps(%)', 'Grs(%)', 'And(%)', 'Uva(%)']})
    factor = 12.0 / total_oxygen
    apfu = {cation: prop * factor for cation, prop in cation_proportions.items()}
    Alm, Pyr, Sps, Grs, And, Uva = 0, 0, 0, 0, 0, 0
    fe_total = apfu.get('Fe', 0);
    mg = apfu.get('Mg', 0);
    mn = apfu.get('Mn', 0);
    ca = apfu.get('Ca', 0);
    al = apfu.get('Al', 0);
    cr = apfu.get('Cr', 0);
    ti = apfu.get('Ti', 0)
    fe3 = max(0, 2.0 - (al + cr + ti));
    fe2 = max(0, fe_total - fe3)
    pyralspite_cations = fe2 + mg + mn
    if pyralspite_cations > 0:
        pyralspite_potential = pyralspite_cations / 3.0
        al_for_pyralspite = min(al / 2.0, pyralspite_potential)
        if al_for_pyralspite > 0:
            Pyr = (mg / pyralspite_cations) * al_for_pyralspite * 3.0;
            Alm = (fe2 / pyralspite_cations) * al_for_pyralspite * 3.0;
            Sps = (mn / pyralspite_cations) * al_for_pyralspite * 3.0
    al_used = (Pyr + Alm + Sps) * 2.0 / 3.0;
    al_rem = al - al_used;
    rem_ca = ca
    if al_rem > 0: Grs = min(al_rem * 1.5, rem_ca); rem_ca -= Grs
    if cr > 0 and rem_ca > 0: Uva = min(cr * 1.5, rem_ca); rem_ca -= Uva
    if fe3 > 0 and rem_ca > 0: And = min(fe3 * 1.5, rem_ca)
    total = Alm + Pyr + Sps + Grs + And + Uva
    if total == 0: total = 1e-9
    return pd.Series({'Alm(%)': (Alm / total) * 100, 'Pyr(%)': (Pyr / total) * 100, 'Sps(%)': (Sps / total) * 100,
                      'Grs(%)': (Grs / total) * 100, 'And(%)': (And / total) * 100, 'Uva(%)': (Uva / total) * 100})


def clean_column_names(df):
    clean_map = {}
    for col in df.columns:
        new_col = str(col).strip()
        for suffix in ['(wt%)', 'wt%', '(%)', '%', '[wt%]', '[%]']: new_col = new_col.replace(suffix, '').strip()
        clean_map[col] = new_col.replace(' ', '')
    return df.rename(columns=clean_map)


def render_analysis_results(res, garnet_features, unique_key=""):
    tab1, tab2 = st.tabs(["📊 主要预测结果", "🔬 石榴石专家分析"])
    with tab1:
        st.subheader("整体矿物识别结果")
        mineral_options = ['全部'] + list(res['Pred_Mineral'].unique())
        selected_minerals = st.multiselect("筛选查看特定矿物：", options=mineral_options, default='全部',
                                           key=f"ms_{unique_key}")
        display_df = res if ('全部' in selected_minerals or not selected_minerals) else res[
            res['Pred_Mineral'].isin(selected_minerals)]
        st.dataframe(display_df, use_container_width=True)
        st.download_button('下载主要预测结果 (CSV)', res.to_csv(index=False).encode('utf-8-sig'),
                           f'mineral_predictions_{unique_key}.csv', mime='text/csv', key=f"dl_gen_{unique_key}")
        st.bar_chart(res['Pred_Mineral'].value_counts())
    with tab2:
        st.markdown("本模块基于**二级专家模型**，对石榴石的**具体亚类**进行智能预测与端元可视化。")
        if garnet_features is not None and not garnet_features.empty:
            st.success(f"🔍 **成功筛选出 {len(garnet_features)} 条石榴石样本，已完成亚类定名与端元计算！**")
            endmember_cols = ['Alm(%)', 'Pyr(%)', 'Sps(%)', 'Grs(%)', 'And(%)', 'Uva(%)']
            endmember_cols_display = [col for col in endmember_cols if
                                      col in garnet_features.columns and garnet_features[col].sum() > 0.01]
            cols_to_show = [c for c in (['PointID', 'Pred_Subtype'] + [c for c in MOLAR_MASSES if
                                                                       c in garnet_features.columns] + endmember_cols_display)
                            if c in garnet_features.columns]
            st.dataframe(garnet_features.get(cols_to_show, garnet_features.columns).style.format("{:.2f}",
                                                                                                 subset=endmember_cols_display),
                         use_container_width=True)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.bar_chart(garnet_features['Pred_Subtype'].value_counts())
            with col_c2:
                st.bar_chart(garnet_features[endmember_cols_display])
            st.download_button('下载石榴石分析结果', garnet_features.to_csv(index=False).encode('utf-8-sig'),
                               f'garnet_analysis_{unique_key}.csv', mime='text/csv', key=f"dl_gar_{unique_key}")
        else:
            st.info("ℹ️ 本次分析中没有检测到石榴石样本，跳过二级专家模型定名。")


# ---------- 5. 加载机器学习模型 ----------
@st.cache_resource(show_spinner="正在加载模型架构...")
def load_artifacts(_refresh_trigger=0):
    try:
        return joblib.load(GENERAL_MODEL_PATH), joblib.load(GENERAL_COLS_PATH), joblib.load(
            GARNET_MODEL_PATH), joblib.load(GARNET_COLS_PATH)
    except Exception as e:
        st.error(f"加载模型文件时发生错误: {e}");
        return None, None, None, None


# 使用 trigger 强制刷新缓存
if 'model_version' not in st.session_state: st.session_state.model_version = 0
general_mdl, general_REQ_COLS, garnet_mdl, garnet_REQ_COLS = load_artifacts(st.session_state.model_version)

# ---------- 6. 侧边栏导航与状态管理 ----------
with st.sidebar:
    st.title(" 系统控制台")
    app_mode = st.radio("请选择工作模式：", ["🔬 矿物智能识别", "🧬 模型进化中心"], index=0)
    st.markdown("---")

    if app_mode == "🔬 矿物智能识别":
        if st.button(" 完全清空历史缓存", type="primary", use_container_width=True):
            clear_all_history();
            st.session_state.clear();
            st.success("数据已清空！");
            time.sleep(1);
            st.rerun()
        st.subheader("📂 历史预测查询")
        try:
            history_df = get_history_list()
            selected_hist = st.selectbox("选择记录查看", options=['无'] + history_df.apply(
                lambda row: f"[{row['id']}] {row['timestamp']}", axis=1).tolist() if not history_df.empty else ['无'])
        except Exception:
            selected_hist = '无'

# ---------- 7. 页面 A：传统智能识别工作流 ----------
if app_mode == "🔬 矿物智能识别":
    st.title('💎 EPMA 矿物智能识别与分析系统')
    st.markdown('---')

    if selected_hist != '无':
        st.markdown("###  历史预测结果展示")
        selected_id = int(selected_hist.split("]")[0].replace("[", ""))
        conn = sqlite3.connect(DB_PATH)
        query_res = pd.read_sql_query(f"SELECT filename, timestamp, data_json FROM history WHERE id={selected_id}",
                                      conn)
        conn.close()
        if not query_res.empty:
            st.info(f"正在查看历史文件: **{query_res.iloc[0]['filename']}**");
            st.warning("💡 如需上传新数据进行分析，请在左侧菜单将历史记录选回【无】。")
            raw_data = json.loads(query_res.iloc[0]['data_json'])
            hist_general = pd.DataFrame(raw_data) if isinstance(raw_data, list) else pd.DataFrame(
                raw_data.get("general", []))
            hist_garnet = pd.DataFrame(raw_data.get("garnet")) if not isinstance(raw_data, list) and raw_data.get(
                "garnet") else None
            render_analysis_results(hist_general, hist_garnet, unique_key=f"hist_{selected_id}")
        st.stop()

    if all([general_mdl, general_REQ_COLS, garnet_mdl, garnet_REQ_COLS]):
        st.markdown("#### 📁 数据上传与预处理")
        uploaded = st.file_uploader('请上传您的 EPMA csv 或 excel 文件进行智能鉴定', type=['csv', 'xlsx'])

        if uploaded:
            file_hash = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.processed_file != file_hash:
                status_msg = st.empty()
                try:
                    df_raw = clean_column_names(
                        pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded))
                    for col in df_raw.columns:
                        if col not in ['PointID', 'Sample']: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
                    df_cleaned = df_raw.fillna(0)
                except Exception as e:
                    status_msg.error(f"读取出错: {e}"); st.stop()

                my_bar = status_msg.progress(0, text="正在通过一级通用模型进行推理...")
                for i in range(100): time.sleep(0.005); my_bar.progress(i + 1, text="正在通过一级通用模型进行推理...")

                X_general = df_cleaned.reindex(columns=general_REQ_COLS, fill_value=0)
                res = df_cleaned.copy()
                res['Pred_Mineral'] = general_mdl.predict(X_general)

                garnet_features = None
                if 'Garnet' in res['Pred_Mineral'].unique():
                    garnet_data = res[res['Pred_Mineral'] == 'Garnet'].copy()
                    status_msg.info(f"🔍 筛选出 {len(garnet_data)} 条石榴石，启动二级专家模型...")
                    time.sleep(0.5)
                    endmember_results = garnet_data.apply(calculate_garnet_formula, axis=1)
                    garnet_features = pd.concat([garnet_data, endmember_results], axis=1)
                    garnet_features['Pred_Subtype'] = garnet_mdl.predict(
                        garnet_features.reindex(columns=garnet_REQ_COLS, fill_value=0))
                else:
                    status_msg.info("ℹ️ 未检测到石榴石样本，跳过专家模型。");
                    time.sleep(0.5)

                save_history(uploaded.name, res, garnet_features)
                st.session_state.processed_file = file_hash
                st.session_state.general_res, st.session_state.garnet_res = res, garnet_features
                status_msg.empty();
                st.success("✅ 分析完成！结果已自动保存。")

            render_analysis_results(st.session_state.general_res, st.session_state.garnet_res, unique_key="current")

# ---------- 8. 页面 B：模型进化中心 (MLOps) ----------
elif app_mode == "🧬 模型进化中心":
    st.title('模型持续进化中心')
    st.markdown("此模块允许您上传带有真实标签的新数据。系统将融合历史主数据重新训练算法，实现预测能力的自我成长。")
    st.markdown('---')

    col_v1, col_v2, col_v3 = st.columns(3)
    user_df = get_all_user_training_data()
    col_v1.metric("出厂核心样本数", "50000+")
    col_v2.metric("当前累积学习新样本", f"{len(user_df)} 条")
    col_v3.metric("模型当前版本状态", "已修改" if len(user_df) > 0 else "出厂默认版")

    st.markdown("#### 💡投喂新知识")
    st.info(" 请上传包含 **`Mineral`** (真实矿物名称) 列的清洗后 CSV 文件。列名需与探针数据元素一致。")
    train_upload = st.file_uploader("上传新增训练数据", type=['csv'])

    if train_upload:
        if st.button("📥 确认投喂并执行增量训练", type="primary"):
            new_train_df = pd.read_csv(train_upload)
            if 'Mineral' not in new_train_df.columns:
                st.error("❌ 上传的文件中找不到 `Mineral` 列，模型无法进行监督学习！")
            else:
                with st.spinner("🧠 正在融合数据并重建随机森林神经网络 (这可能需要几十秒)..."):
                    try:
                        # 1. 先不写数据库！在内存中合并主数据、已有的历史增量数据、以及本次的新数据
                        df_master = pd.read_csv(DATA_PATH)
                        df_all_user = get_all_user_training_data()
                        df_combined = pd.concat([df_master, df_all_user, new_train_df], ignore_index=True)

                        # 2. 按照原本的 SMOTE 逻辑进行预处理
                        df_combined = df_combined.groupby('Mineral').filter(lambda x: len(x) > 1)
                        X = df_combined.drop(columns=['PointID', 'Mineral'], errors='ignore')
                        X = clean_column_names(X)
                        y = df_combined['Mineral']

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                            stratify=y)

                        # 动态 SMOTE 补齐
                        k_neighbors_max = 5
                        counts_train = y_train.value_counts()
                        under_represented_classes = counts_train[counts_train <= k_neighbors_max].index
                        if not under_represented_classes.empty:
                            for cls in under_represented_classes:
                                num_samples = counts_train[cls]
                                indices_to_add = np.random.choice(y_train[y_train == cls].index,
                                                                  size=(k_neighbors_max + 1 - num_samples),
                                                                  replace=True)
                                X_train = pd.concat([X_train, X_train.loc[indices_to_add]], ignore_index=True)
                                y_train = pd.concat([y_train, y_train.loc[indices_to_add]], ignore_index=True)

                        k_neighbors_for_smote = min(k_neighbors_max, y_train.value_counts().min() - 1)
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_for_smote)
                        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

                        # 3. 训练新模型
                        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
                        clf.fit(X_train_sm, y_train_sm)

                        # 4. [核心修复点] 只有模型没有报错且训练到底了，才正式落盘数据并备份替换！
                        save_user_training_data(new_train_df)  # 正式写入 SQLite
                        shutil.copy(GENERAL_MODEL_PATH, PREV_MODEL_PATH)  # 备份上一代模型

                        joblib.dump(clf, GENERAL_MODEL_PATH)  # 覆盖新模型
                        joblib.dump(X_train.columns.tolist(), GENERAL_COLS_PATH)

                        st.session_state.model_version += 1  # 强制刷新页面缓存
                        st.success(
                            f"🎉 进化成功！模型已吸收 {len(new_train_df)} 条新知识。新的验证集准确率: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"训练过程中发生错误: {e}")
                        # [提示优化] 明确告诉用户脏数据没有被保存
                        st.info("⚠️ 您的数据没有被录入系统，原模型未受影响；请检查数据格式是否规范")

    st.markdown("#### 🛠️ 模型版本控制台")
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button(" 撤销上一轮训练 (回退节点)", use_container_width=True):
            if os.path.exists(PREV_MODEL_PATH):
                shutil.copy(PREV_MODEL_PATH, GENERAL_MODEL_PATH)
                delete_last_training_batch()
                st.session_state.model_version += 1
                st.success("已成功回退到上一次的模型状态！");
                time.sleep(1);
                st.rerun()
            else:
                st.warning("没有找到上一版本的备份文件。")
    with cc2:
        if st.button(" 恢复出厂设置 (清除所有学习记忆)", type="primary", use_container_width=True):
            if os.path.exists(FACTORY_MODEL_PATH):
                shutil.copy(FACTORY_MODEL_PATH, GENERAL_MODEL_PATH)
                clear_user_training_data()
                st.session_state.model_version += 1
                st.success("已清除所有用户灌输的数据，模型恢复为出厂纯净版！");
                time.sleep(1.5);
                st.rerun()
            else:
                st.error("出厂模型文件缺失，无法恢复！")

st.markdown('---')
st.caption('🛠️  Powered by Hierarchical Reasoning Models | 技术支持: 1003881004@qq.com 刘沅杭')
