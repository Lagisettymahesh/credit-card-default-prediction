# ==========================================
# Credit Card Default Prediction - Streamlit
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Credit Default Predictor", layout="wide")

st.title("💳 Credit Card Default Prediction")

st.markdown("Upload the **UCI Credit Card dataset** to train ML models and compare performance.")

# ==========================================
# Upload Dataset
# ==========================================

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

    if "ID" in df.columns:
        df.drop(columns=['ID'], inplace=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Shape:", df.shape)
    st.write("Target Distribution")
    st.write(df['default'].value_counts())

    # ==========================================
    # EDA
    # ==========================================

    st.subheader("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots()
        df['default'].value_counts().plot.pie(
            autopct='%1.1f%%',
            labels=['No Default','Default'],
            colors=['#2ecc71','#e74c3c'],
            ax=ax1
        )
        ax1.set_title("Default Distribution")
        st.pyplot(fig1)

    with col2:
        cols = ['LIMIT_BAL','AGE','PAY_0','BILL_AMT1','PAY_AMT1','default']
        fig2, ax2 = plt.subplots()
        sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("Correlation Heatmap")
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        df.groupby('default')['LIMIT_BAL'].mean().plot.bar(
            color=['#2ecc71','#e74c3c'], ax=ax3
        )
        ax3.set_xticklabels(['No Default','Default'], rotation=0)
        ax3.set_title("Avg Credit Limit by Default")
        st.pyplot(fig3)

    # ==========================================
    # Train Models
    # ==========================================

    if st.button("Train Models"):

        # Fix unknown categories
        df['EDUCATION'] = df['EDUCATION'].replace({0:4,5:4,6:4})
        df['MARRIAGE']  = df['MARRIAGE'].replace({0:3})

        X = df.drop(columns=['default'])
        y = df['default']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2,
            random_state=42, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        results = {}

        st.subheader("Model Results")

        for name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

            auc = roc_auc_score(y_test, y_prob)

            results[name] = auc

            st.write(f"### {name}")
            st.text(classification_report(y_test, y_pred))
            st.write("ROC AUC:", round(auc,4))

        # ==========================================
        # Model Comparison
        # ==========================================

        st.subheader("Model Comparison")

        fig4, ax4 = plt.subplots()

        ax4.bar(results.keys(), results.values(),
                color=['#3498db','#2ecc71','#e74c3c'])

        ax4.set_ylim(0.5,1)
        ax4.set_ylabel("ROC AUC")
        ax4.set_title("Model Comparison")

        for i,(k,v) in enumerate(results.items()):
            ax4.text(i,v+0.005,round(v,3),ha='center')

        st.pyplot(fig4)

        best = max(results, key=results.get)

        st.success(f"Best Model: {best} (AUC = {round(results[best],4)})")
