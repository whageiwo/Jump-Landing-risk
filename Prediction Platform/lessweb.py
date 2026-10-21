import streamlit as st
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="LESS Risk Prediction Platform", layout="wide")

# ------------------ 全局字体 ------------------
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# ------------------ 页面标题 ------------------
st.markdown("<h1 style='text-align: center; color: darkred;'>LESS Risk Prediction Platform</h1>", unsafe_allow_html=True)

# ------------------ 加载模型 ------------------
model = xgb.XGBClassifier()
model.load_model("optimized_xgboost_model.bin")

# ------------------ 定义特征名称 ------------------
feature_names = [
    "KF-IC", "HF-IC", "TF-IC", "TLF-IC", "APF-IC",
    "KMD-IC", "BOSW", "FP", "FCS", "KFD",
    "HFD", "TFD", "RKV"
]

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 左列 7 个特征 --------
with col1:
    for name in feature_names[:7]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        if name in ["KMD-IC", "FCS", "RKV"]:
            val = st.radio("", options=[0,1], key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# -------- 中列 6 个特征 --------
with col2:
    for name in feature_names[7:]:
        st.markdown(f"<p style='font-size:{label_size}; margin:0'>{name}</p>", unsafe_allow_html=True)
        if name in ["KMD-IC", "FCS", "RKV"]:
            val = st.radio("", options=[0,1], key=name)
        else:
            val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测结果放在第二列特征下面 --------
pred = model.predict(X_input)
proba = model.predict_proba(X_input)[0][1]

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:darkgreen;'>Predicted Probability</h3>", unsafe_allow_html=True)
    if pred[0] == 1:
        st.markdown(f"<p style='color:red; font-size:40px; font-weight:bold;'>{proba:.2f} (High Risk)</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:blue; font-size:40px; font-weight:bold;'>{proba:.2f} (Low Risk)</p>", unsafe_allow_html=True)

# -------- 右列：SHAP 可视化 --------
with col3:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    shap_expl = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_input[0],
        feature_names=feature_names
    )

    # 瀑布图
    st.markdown("<h3 style='color:darkorange;'>Waterfall Plot</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,6))
    shap.plots.waterfall(shap_expl, show=False)
    st.pyplot(fig)

    # 力图 (Streamlit 显示)
    st.markdown("<h3 style='color:purple;'>Force Plot</h3>", unsafe_allow_html=True)
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values.values[0], X_input[0], feature_names=feature_names
    )
    components.html(f"<head>{shap.getjs()}</head>{force_plot.html()}", height=300) 
