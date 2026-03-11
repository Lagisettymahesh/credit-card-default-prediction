# ==========================================
#   Credit Card Default Prediction - SMALL
# ==========================================

# STEP 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==========================================
# STEP 2: Load Data
# ==========================================
df = pd.read_csv('UCI_Credit_Card.csv')
df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)
df.drop(columns=['ID'], inplace=True)

print("Shape:", df.shape)
print(df['default'].value_counts())

# ==========================================
# STEP 3: EDA - Quick Plots
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Pie chart
df['default'].value_counts().plot.pie(
    ax=axes[0], autopct='%1.1f%%',
    labels=['No Default', 'Default'],
    colors=['#2ecc71', '#e74c3c']
)
axes[0].set_title('Target Distribution')

# Correlation heatmap
cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1', 'default']
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', ax=axes[1])
axes[1].set_title('Correlation Heatmap')

# Credit limit by default
df.groupby('default')['LIMIT_BAL'].mean().plot.bar(
    ax=axes[2], color=['#2ecc71', '#e74c3c']
)
axes[2].set_title('Avg Credit Limit by Default')
axes[2].set_xticklabels(['No Default', 'Default'], rotation=0)

plt.tight_layout()
plt.savefig('eda.png')
plt.show()
print("EDA saved!")

# ==========================================
# STEP 4: Preprocessing
# ==========================================
# Fix unknown categories
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE']  = df['MARRIAGE'].replace({0: 3})

X = df.drop(columns=['default'])
y = df['default']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# STEP 5: Train-Test Split + SMOTE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:", dict(pd.Series(y_train).value_counts()))

# ==========================================
# STEP 6: Train 3 Models
# ==========================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    results[name] = auc
    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", round(auc, 4))

# ==========================================
# STEP 7: Compare Models
# ==========================================
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), results.values(),
        color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Model Comparison - ROC AUC')
plt.ylabel('ROC AUC Score')
plt.ylim(0.5, 1.0)
for i, (k, v) in enumerate(results.items()):
    plt.text(i, v + 0.005, str(round(v, 3)), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('comparison.png')
plt.show()

best = max(results, key=results.get)
print(f"\n Best Model: {best} → AUC = {round(results[best], 4)}")
