import streamlit as st
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.text import Text
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(
    page_title="LESS 风险预测平台",
    layout="wide"
)

# ------------------ 全局字体 ------------------
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"

# ------------------ SHAP 图表配色 ------------------
SHAP_POS_COLOR = "#F28C8C"  # 正向贡献：浅红色
SHAP_NEG_COLOR = "#7CB5EC"  # 负向贡献：浅蓝色


def recolor_waterfall(fig, pos_color, neg_color):
    """仅替换瀑布图中的红蓝色块、边框及对应彩色文字。"""

    def replacement_color(color):
        try:
            r, g, b, a = mcolors.to_rgba(color)
        except (TypeError, ValueError):
            return None

        if a < 0.05:
            return None

        if r > b and r > 0.35:
            return pos_color

        if b > r and b > 0.35:
            return neg_color

        return None

    for artist in fig.findobj():
        if hasattr(artist, "get_facecolor"):
            color = replacement_color(artist.get_facecolor())

            if color is not None:
                artist.set_facecolor(color)

                if hasattr(artist, "set_edgecolor"):
                    artist.set_edgecolor(color)

        if isinstance(artist, Text):
            color = replacement_color(artist.get_color())

            if color is not None:
                artist.set_color(color)


# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred;'>"
    "LESS 风险预测平台"
    "</h1>",
    unsafe_allow_html=True
)

# ------------------ 加载模型 ------------------
model = xgb.XGBClassifier()
model.load_model("optimized_xgboost_model.bin")

# ------------------ 定义特征名称 ------------------
# 保留原始特征顺序和简写，供模型预测和 SHAP 图表使用。
feature_names = [
    "KF-IC", "HF-IC", "TF-IC", "TLF-IC", "APF-IC",
    "KMD-IC", "BOSW", "FP", "FCS", "KFD",
    "HFD", "TFD", "RKV"
]

# 输入框标签：简写（中文名称）
# SHAP 图表中的特征名称继续使用英文简写。
feature_labels = {
    "KF-IC": "KF-IC（膝关节屈曲-初始接触）",
    "HF-IC": "HF-IC（髋关节屈曲-初始接触）",
    "TF-IC": "TF-IC（躯干屈曲-初始接触）",
    "TLF-IC": "TLF-IC（躯干侧屈-初始接触）",
    "APF-IC": "APF-IC（踝关节跖屈-初始接触）",
    "KMD-IC": "KMD-IC（膝关节内移-初始接触）",
    "BOSW": "BOSW（支撑面宽度）",
    "FP": "FP（足部位置）",
    "FCS": "FCS（足部接触对称性）",
    "KFD": "KFD（膝关节屈曲位移）",
    "HFD": "HFD（髋关节屈曲位移）",
    "TFD": "TFD（躯干屈曲位移）",
    "RKV": "RKV（膝关节外翻活动范围）",
}

# ------------------ 页面布局 ------------------
col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
label_size = "16px"
inputs = []

# -------- 左列 7 个特征 --------
with col1:
    for name in feature_names[:7]:
        st.markdown(
            f"<p style='font-size:{label_size}; margin:0'>"
            f"{feature_labels[name]}</p>",
            unsafe_allow_html=True
        )

        if name in ["KMD-IC", "FCS", "RKV"]:
            val = st.radio(
                "",
                options=[0, 1],
                key=name
            )
        else:
            val = st.number_input(
                "",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=name
            )

        inputs.append(val)

# -------- 中列 6 个特征 --------
with col2:
    for name in feature_names[7:]:
        st.markdown(
            f"<p style='font-size:{label_size}; margin:0'>"
            f"{feature_labels[name]}</p>",
            unsafe_allow_html=True
        )

        if name in ["KMD-IC", "FCS", "RKV"]:
            val = st.radio(
                "",
                options=[0, 1],
                key=name
            )
        else:
            val = st.number_input(
                "",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=name
            )

        inputs.append(val)

X_input = np.array([inputs])

# -------- 预测结果放在第二列特征下面 --------
pred = model.predict(X_input)
proba = model.predict_proba(X_input)[0][1]

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='color:darkgreen;'>预测概率</h3>",
        unsafe_allow_html=True
    )

    if pred[0] == 1:
        st.markdown(
            "<p style='color:red; font-size:40px; font-weight:bold;'>"
            f"{proba:.2f}（高风险）</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='color:blue; font-size:40px; font-weight:bold;'>"
            f"{proba:.2f}（低风险）</p>",
            unsafe_allow_html=True
        )

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

    # -------- 瀑布图 --------
    st.markdown(
        "<h3 style='color:darkorange;'>SHAP 瀑布图</h3>",
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    shap.plots.waterfall(
        shap_expl,
        show=False
    )

    # 仅调整图表颜色
    recolor_waterfall(
        fig,
        SHAP_POS_COLOR,
        SHAP_NEG_COLOR
    )

    st.pyplot(fig)

    # -------- 力图（Streamlit 显示）--------
    st.markdown(
        "<h3 style='color:purple;'>SHAP 力图</h3>",
        unsafe_allow_html=True
    )

    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_input[0],
        feature_names=feature_names,
        # 顺序：正向贡献颜色、负向贡献颜色
        plot_cmap=[SHAP_POS_COLOR, SHAP_NEG_COLOR]
    )

    components.html(
        f"<head>{shap.getjs()}</head>{force_plot.html()}",
        height=300
    )
