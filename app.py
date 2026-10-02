# ============================================================
# MULTI-CLASS EEG SIGNAL CLASSIFICATION USING MACHINE LEARNING
# Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EEG Signal Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "rf_model.pkl"
GRAPHS_DIR = BASE_DIR / "graphs"

GAUSSIAN_CSV = BASE_DIR / "gaussian_noise_results.csv"
IMPULSE_CSV = BASE_DIR / "impulse_noise_results.csv"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    f"X{i}"
    for i in range(1, 17)
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL BACKGROUND
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.09),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, 0.07),
                transparent 25%
            ),
            #080d18;
    }

    .main {
        background-color: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }


    /* ======================================================
       HEADINGS
    ====================================================== */

    h1 {
        color: #f8fafc !important;
        font-weight: 750 !important;
        letter-spacing: -0.6px;
    }

    h2 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #e2e8f0 !important;
        font-weight: 650 !important;
    }

    p {
        color: #cbd5e1;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #070b14 0%,
                #0b1120 100%
            );

        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] p {
        color: #cbd5e1;
    }


    /* ======================================================
       METRICS
    ====================================================== */

    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #26344a;
        border-radius: 16px;
        padding: 18px 20px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 750 !important;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        width: 100%;
        min-height: 45px;

        border-radius: 11px;

        font-weight: 650;

        border: 1px solid #334155;

        transition:
            all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        transform: translateY(-1px);
    }


    /* ======================================================
       INPUT BOXES
    ====================================================== */

    div[data-baseweb="input"] {
        background-color: #0f172a;
        border-radius: 9px;
    }


    /* ======================================================
       SELECT BOXES
    ====================================================== */

    div[data-baseweb="select"] {
        background-color: #0f172a;
        border-radius: 9px;
    }


    /* ======================================================
       DATAFRAMES
    ====================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #26344a;
        border-radius: 12px;
        overflow: hidden;
    }


    /* ======================================================
       EXPANDERS
    ====================================================== */

    div[data-testid="stExpander"] {
        background-color: rgba(15, 23, 42, 0.65);

        border: 1px solid #26344a;

        border-radius: 12px;
    }


    /* ======================================================
       DIVIDER
    ====================================================== */

    .section-divider {
        height: 1px;
        background-color: #1e293b;
        margin: 30px 0;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;

        color: #64748b;

        padding-top: 40px;

        font-size: 0.82rem;
    }


    /* ======================================================
       RESULT ANCHOR
    ====================================================== */

    #classification-results {
        scroll-margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(
        MODEL_PATH
    )


rf_model = load_model()


# ============================================================
# MODEL ERROR HANDLING
# ============================================================

if rf_model is None:

    st.error(
        "⚠️ Random Forest model could not be found."
    )

    st.markdown(
        f"""
        The application expects the trained model at:

        `{MODEL_PATH}`

        Your project should contain:

        ```text
        models/
        └── rf_model.pkl
        ```
        """
    )

    st.stop()


# ============================================================
# PROBABILITY CHART FUNCTION
# ============================================================

def create_probability_plot(
    probability_df
):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    bars = ax.bar(
        probability_df["Class"],
        probability_df["Probability (%)"]
    )

    ax.set_title(
        "Prediction Probability by EEG Class",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel(
        "EEG Class"
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    maximum = (
        probability_df[
            "Probability (%)"
        ].max()
    )

    ax.set_ylim(
        0,
        max(
            100,
            maximum + 12
        )
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height + 1,

            f"{height:.1f}%",

            ha="center",

            va="bottom",

            fontsize=9
        )

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    # 🧠 EEG Classifier

    **Machine Learning Analysis System**
    """
)

st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",

    [
        "🏠 Overview",
        "🔬 Classify EEG",
        "📊 Model Analysis",
        "🔊 Noise Analysis"
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    "### Project Information"
)

st.sidebar.write(
    "**Dataset:** BEED"
)

st.sidebar.write(
    "**Features:** 16 EEG features"
)

st.sidebar.write(
    "**Classes:** 4"
)

st.sidebar.write(
    "**Final Model:** Random Forest"
)

st.sidebar.write(
    "**Test Accuracy:** 95.67%"
)


st.sidebar.markdown("---")


st.sidebar.caption(
    "Multi-Class EEG Signal Classification "
    "Using Machine Learning"
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title(
        "🧠 EEG Signal Classification"
    )

    st.caption(
        "Multi-Class EEG Signal Classification Using Machine Learning"
    )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Final Test Accuracy",
        "95.67%"
    )


    col2.metric(
        "EEG Features",
        "16"
    )


    col3.metric(
        "Classes",
        "4"
    )


    col4.metric(
        "Final Model",
        "Random Forest"
    )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # ABOUT PROJECT
    # --------------------------------------------------------

    st.subheader(
        "📌 About the Project"
    )


    st.write(
        """
        This project focuses on **multi-class EEG signal
        classification** using machine learning techniques.

        The **BEED dataset** is used as the primary data source,
        containing **16 input EEG features** and **4 target
        classes**.

        Four machine learning algorithms were evaluated:

        **Logistic Regression, K-Nearest Neighbors,
        Random Forest, and Support Vector Machine.**
        """
    )


    st.info(
        "Random Forest achieved a final test accuracy of "
        "**95.67%** on the unseen test dataset."
    )


    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader(
        "⚙️ System Workflow"
    )


    w1, w2, w3, w4 = st.columns(4)


    with w1:

        st.markdown("### 01")

        st.write(
            "**EEG Input**"
        )

        st.caption(
            "16 numerical EEG features."
        )


    with w2:

        st.markdown("### 02")

        st.write(
            "**Feature Arrangement**"
        )

        st.caption(
            "Values are organized as X1–X16."
        )


    with w3:

        st.markdown("### 03")

        st.write(
            "**Random Forest**"
        )

        st.caption(
            "The trained classifier processes the input."
        )


    with w4:

        st.markdown("### 04")

        st.write(
            "**Classification**"
        )

        st.caption(
            "One of four EEG classes is predicted."
        )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROJECT HIGHLIGHTS
    # --------------------------------------------------------

    st.subheader(
        "✨ Project Highlights"
    )


    h1, h2, h3 = st.columns(3)


    with h1:

        st.info(
            "**Model Comparison**\n\n"
            "Four machine learning algorithms "
            "were evaluated."
        )


    with h2:

        st.info(
            "**Noise Analysis**\n\n"
            "Gaussian and impulse noise were "
            "tested at different levels."
        )


    with h3:

        st.info(
            "**Final Evaluation**\n\n"
            "Random Forest achieved 95.67% "
            "accuracy on unseen test data."
        )


# ============================================================
# PAGE 2 — CLASSIFY EEG
# ============================================================

elif page == "🔬 Classify EEG":

    st.title(
        "🔬 Classify EEG Signal"
    )

    st.caption(
        "Enter the 16 EEG feature values and generate a "
        "machine learning prediction."
    )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    st.info(
        "The Random Forest model expects **16 numerical inputs** "
        "from X1 through X16. Enter values using the same feature "
        "format used during model training."
    )


    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.subheader(
        "🧾 EEG Feature Input"
    )


    st.caption(
        "Enter the values for each EEG feature."
    )


    input_values = {}


    input_columns = st.columns(4)


    for index, feature in enumerate(FEATURES):

        with input_columns[index % 4]:

            input_values[feature] = st.number_input(
                feature,

                value=0.0,

                format="%.4f",

                key=f"eeg_input_{feature}"
            )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [input_values],

        columns=FEATURES
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Analyze EEG Signal",

        type="primary",

        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            prediction = rf_model.predict(
                input_df
            )


            probabilities = (
                rf_model.predict_proba(
                    input_df
                )[0]
            )


        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(
                error
            )

            st.stop()


        predicted_class = prediction[0]


        confidence = (
            np.max(probabilities)
            * 100
        )


        # ----------------------------------------------------
        # RESULT ANCHOR
        # ----------------------------------------------------

        st.markdown(
            '<div id="classification-results"></div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="section-divider"></div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # AUTO SCROLL
        # ----------------------------------------------------

        st.markdown(
            """
            <script>

            setTimeout(function() {

                const resultSection =
                    window.parent.document.getElementById(
                        "classification-results"
                    );

                if (resultSection) {

                    resultSection.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });

                }

            }, 350);

            </script>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # CLASSIFICATION RESULT
        # ----------------------------------------------------

        st.subheader(
            "🎯 Classification Result"
        )


        result1, result2, result3 = st.columns(3)


        result1.metric(
            "Predicted Class",
            f"Class {predicted_class}"
        )


        result2.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


        result3.metric(
            "Model Used",
            "Random Forest"
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        st.success(
            f"The Random Forest model classified the entered "
            f"EEG feature pattern as **Class {predicted_class}**."
        )


        # ----------------------------------------------------
        # PROBABILITY DATA
        # ----------------------------------------------------

        probability_df = pd.DataFrame(
            {
                "Class": [
                    f"Class {c}"
                    for c in rf_model.classes_
                ],

                "Probability (%)": (
                    probabilities * 100
                ).round(2)
            }
        )


        # ----------------------------------------------------
        # PROBABILITY SECTION
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Probabilities"
        )


        chart_col, table_col = st.columns(
            [2, 1]
        )


        with chart_col:

            probability_fig = (
                create_probability_plot(
                    probability_df
                )
            )


            st.pyplot(
                probability_fig,

                use_container_width=True
            )


            plt.close(
                probability_fig
            )


        with table_col:

            st.write(
                "Probability Details"
            )


            st.dataframe(
                probability_df,

                use_container_width=True,

                hide_index=True
            )


            st.caption(
                "The class with the highest probability "
                "is selected as the model prediction."
            )


        # ----------------------------------------------------
        # INPUT FEATURE GRAPH
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-divider"></div>',
            unsafe_allow_html=True
        )


        st.subheader(
            "📈 Input EEG Feature Pattern"
        )


        feature_fig, feature_ax = plt.subplots(
            figsize=(12, 4)
        )


        feature_ax.plot(
            FEATURES,

            input_df.iloc[0].values,

            marker="o",

            linewidth=2
        )


        feature_ax.set_title(
            "Entered EEG Feature Values"
        )


        feature_ax.set_xlabel(
            "EEG Features"
        )


        feature_ax.set_ylabel(
            "Feature Value"
        )


        feature_ax.grid(
            True,

            alpha=0.2
        )


        feature_ax.tick_params(
            axis="x",

            rotation=45
        )


        feature_fig.tight_layout()


        st.pyplot(
            feature_fig,

            use_container_width=True
        )


        plt.close(
            feature_fig
        )


        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        with st.expander(
            "ℹ️ Understand this prediction"
        ):

            st.write(
                f"""
                The trained Random Forest classifier predicted
                **Class {predicted_class}** for the entered EEG
                feature pattern.

                The highest model probability was
                **{confidence:.2f}%**.

                These probabilities represent the output of the
                machine learning classifier. They should not be
                interpreted as a medical diagnosis or as a direct
                probability of a medical condition.
                """
            )


# ============================================================
# PAGE 3 — MODEL ANALYSIS
# ============================================================

elif page == "📊 Model Analysis":

    st.title(
        "📊 Model Analysis"
    )


    st.caption(
        "Comparison and analysis of the machine learning models "
        "evaluated during the project."
    )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL RESULTS
    # --------------------------------------------------------

    model_results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "KNN",
                "Random Forest",
                "SVM"
            ],

            "Validation Accuracy (%)": [
                47.08,
                95.58,
                96.83,
                75.25
            ],

            "Macro F1 Score": [
                0.48,
                0.96,
                0.97,
                0.74
            ]
        }
    )


    st.subheader(
        "🤖 Model Comparison"
    )


    st.dataframe(
        model_results,

        use_container_width=True,

        hide_index=True
    )


    # --------------------------------------------------------
    # ACCURACY CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Validation Accuracy Comparison"
    )


    model_fig, model_ax = plt.subplots(
        figsize=(10, 4.5)
    )


    bars = model_ax.bar(
        model_results["Model"],

        model_results[
            "Validation Accuracy (%)"
        ]
    )


    model_ax.set_ylabel(
        "Accuracy (%)"
    )


    model_ax.set_ylim(
        0,

        100
    )


    model_ax.set_title(
        "Machine Learning Model Performance"
    )


    model_ax.grid(
        axis="y",

        alpha=0.2
    )


    model_ax.tick_params(
        axis="x",

        rotation=15
    )


    for bar in bars:

        height = bar.get_height()

        model_ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            height + 1,

            f"{height:.2f}%",

            ha="center",

            va="bottom",

            fontsize=9
        )


    model_fig.tight_layout()


    st.pyplot(
        model_fig,

        use_container_width=True
    )


    plt.close(
        model_fig
    )


    # --------------------------------------------------------
    # FINAL MODEL METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "🌲 Final Random Forest Model"
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "Validation Accuracy",
        "96.83%"
    )


    m2.metric(
        "Test Accuracy",
        "95.67%"
    )


    m3.metric(
        "Macro F1 Score",
        "0.97"
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader(
        "🌲 Random Forest Feature Importance"
    )


    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,

            "Importance":
                rf_model.feature_importances_
        }
    )


    importance_df = (
        importance_df
        .sort_values(
            "Importance",

            ascending=True
        )
    )


    importance_fig, importance_ax = plt.subplots(
        figsize=(10, 7)
    )


    importance_ax.barh(
        importance_df["Feature"],

        importance_df["Importance"]
    )


    importance_ax.set_xlabel(
        "Importance"
    )


    importance_ax.set_ylabel(
        "Feature"
    )


    importance_ax.set_title(
        "Random Forest Feature Importance"
    )


    importance_ax.grid(
        axis="x",

        alpha=0.2
    )


    importance_fig.tight_layout()


    st.pyplot(
        importance_fig,

        use_container_width=True
    )


    plt.close(
        importance_fig
    )


    highest_feature = (
        importance_df.iloc[-1]
    )


    st.info(
        f"**{highest_feature['Feature']}** has the highest "
        f"Random Forest feature importance "
        f"({highest_feature['Importance']:.3f})."
    )


    with st.expander(
        "ℹ️ What does feature importance mean?"
    ):

        st.write(
            """
            Feature importance represents the relative contribution
            of each input feature to the Random Forest's decision
            process.

            A higher value indicates that the feature contributed
            more strongly to the model's classification decisions.

            Feature importance alone does not establish a biological
            or physiological meaning for an EEG feature.
            """
        )


# ============================================================
# PAGE 4 — NOISE ANALYSIS
# ============================================================

elif page == "🔊 Noise Analysis":

    st.title(
        "🔊 Noise Robustness Analysis"
    )


    st.caption(
        "Evaluation of model performance under artificial "
        "Gaussian and impulse noise."
    )


    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.info(
        "Noise analysis evaluates how classification accuracy "
        "changes when increasing levels of artificial noise "
        "are introduced into the EEG feature data."
    )


    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    st.subheader(
        "📉 Gaussian Noise"
    )


    if GAUSSIAN_CSV.exists():

        gaussian_df = pd.read_csv(
            GAUSSIAN_CSV
        )


        st.dataframe(
            gaussian_df,

            use_container_width=True,

            hide_index=True
        )


        st.caption(
            "Accuracy measured at different Gaussian "
            "noise levels."
        )


        noise_column = (
            gaussian_df.columns[0]
        )


        accuracy_columns = [
            column

            for column in gaussian_df.columns

            if column != noise_column
        ]


        gaussian_fig, gaussian_ax = plt.subplots(
            figsize=(11, 5)
        )


        for column in accuracy_columns:

            gaussian_ax.plot(
                gaussian_df[noise_column],

                gaussian_df[column],

                marker="o",

                linewidth=2,

                label=column
            )


        gaussian_ax.set_title(
            "Model Performance Under Gaussian Noise"
        )


        gaussian_ax.set_xlabel(
            "Noise Level (%)"
        )


        gaussian_ax.set_ylabel(
            "Accuracy (%)"
        )


        gaussian_ax.grid(
            True,

            alpha=0.2
        )


        gaussian_ax.legend()


        gaussian_fig.tight_layout()


        st.pyplot(
            gaussian_fig,

            use_container_width=True
        )


        plt.close(
            gaussian_fig
        )


    else:

        gaussian_image = (
            GRAPHS_DIR
            / "11_gaussian_noise_performance.png"
        )


        if gaussian_image.exists():

            st.image(
                str(gaussian_image),

                use_container_width=True
            )

        else:

            st.warning(
                "Gaussian noise results were not found."
            )


    # --------------------------------------------------------
    # IMPULSE NOISE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "⚡ Impulse Noise"
    )


    if IMPULSE_CSV.exists():

        impulse_df = pd.read_csv(
            IMPULSE_CSV
        )


        st.dataframe(
            impulse_df,

            use_container_width=True,

            hide_index=True
        )


        st.caption(
            "Accuracy measured at different impulse "
            "noise levels."
        )


        noise_column = (
            impulse_df.columns[0]
        )


        accuracy_columns = [
            column

            for column in impulse_df.columns

            if column != noise_column
        ]


        impulse_fig, impulse_ax = plt.subplots(
            figsize=(11, 5)
        )


        for column in accuracy_columns:

            impulse_ax.plot(
                impulse_df[noise_column],

                impulse_df[column],

                marker="o",

                linewidth=2,

                label=column
            )


        impulse_ax.set_title(
            "Model Performance Under Impulse Noise"
        )


        impulse_ax.set_xlabel(
            "Noise Level (%)"
        )


        impulse_ax.set_ylabel(
            "Accuracy (%)"
        )


        impulse_ax.grid(
            True,

            alpha=0.2
        )


        impulse_ax.legend()


        impulse_fig.tight_layout()


        st.pyplot(
            impulse_fig,

            use_container_width=True
        )


        plt.close(
            impulse_fig
        )


    else:

        impulse_image = (
            GRAPHS_DIR
            / "12_impulse_noise_performance.png"
        )


        if impulse_image.exists():

            st.image(
                str(impulse_image),

                use_container_width=True
            )

        else:

            st.warning(
                "Impulse noise results were not found."
            )


    # --------------------------------------------------------
    # OBSERVATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "🔎 Observations"
    )


    o1, o2, o3 = st.columns(3)


    with o1:

        st.info(
            "**Gaussian Noise**\n\n"
            "KNN showed comparatively strong performance "
            "as Gaussian noise increased."
        )


    with o2:

        st.info(
            "**Impulse Noise**\n\n"
            "Random Forest maintained strong performance "
            "under impulse noise."
        )


    with o3:

        st.info(
            "**Overall Trend**\n\n"
            "Increasing noise generally reduced "
            "classification accuracy."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="section-divider"></div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="footer">

    <b>
    Multi-Class EEG Signal Classification Using Machine Learning
    </b>

    <br><br>

    BEED Dataset
    &nbsp;•&nbsp;
    Random Forest
    &nbsp;•&nbsp;
    Streamlit

    </div>
    """,
    unsafe_allow_html=True
)