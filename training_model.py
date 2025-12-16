import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ==========================================
# STEP 1: Generate Synthetic Dataset (same as before)
# ==========================================
def generate_synthetic_sepsis_data(n_patients=500, max_hours=72, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        hours = rng.integers(24, max_hours + 1)
        age = int(rng.integers(18, 90))
        sex = int(rng.choice([0, 1]))
        baseline_hr, baseline_rr, baseline_temp = rng.normal(80,10), rng.normal(18,3), rng.normal(36.8,0.4)
        baseline_map, baseline_spo2 = rng.normal(85,7), rng.normal(97,1)
        baseline_wbc, baseline_lactate = rng.normal(8,2), rng.normal(1.2,0.5)

        will_sepsis = rng.random() < 0.15
        sepsis_onset = hours - rng.integers(4, 10) if will_sepsis else None

        for t in range(hours):
            hr = baseline_hr + rng.normal(0, 5)
            rr = baseline_rr + rng.normal(0, 2)
            temp = baseline_temp + rng.normal(0, 0.3)
            mapv = baseline_map + rng.normal(0, 4)
            spo2 = baseline_spo2 + rng.normal(0, 0.5)
            wbc = baseline_wbc + rng.normal(0, 1.5)
            lactate = baseline_lactate + rng.normal(0, 0.2)

            if will_sepsis and sepsis_onset and t >= sepsis_onset:
                hr += rng.uniform(10, 40)
                rr += rng.uniform(3, 10)
                temp += rng.uniform(0.5, 2.0)
                mapv -= rng.uniform(5, 20)
                lactate += rng.uniform(1.0, 3.5)
                wbc += rng.uniform(3, 6)

            label = 1 if (will_sepsis and sepsis_onset and t >= sepsis_onset - 6) else 0
            rows.append([pid, t, hr, rr, temp, mapv, spo2, wbc, lactate, age, sex, label])

    cols = ["patient_id", "ts", "HR", "RR", "Temp", "MAP", "SpO2",
            "WBC", "Lactate", "Age", "Sex", "label"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"✅ Generated dataset with {len(df)} rows, {n_patients} patients.")
    return df

# ==========================================
# STEP 2: Prepare Data
# ==========================================
df = generate_synthetic_sepsis_data()

X = df[["HR", "RR", "Temp", "MAP", "SpO2", "WBC", "Lactate", "Age", "Sex"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("✅ Train/Test split done.")

# ==========================================
# STEP 3: Balance Data using SMOTE
# ==========================================
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print(f"✅ After balancing: {sum(y_train_bal==1)} sepsis cases, {sum(y_train_bal==0)} non-sepsis cases")

# ==========================================
# STEP 4: Train XGBoost Model
# ==========================================
print("\n🧠 Training model...")
model = XGBClassifier(
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=1,  # not needed since SMOTE balanced data
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_bal, y_train_bal)
print("✅ Training complete.\n")

# ==========================================
# STEP 5: Evaluate with Adjusted Threshold
# ==========================================
y_proba = model.predict_proba(X_test)[:, 1]

# Tune threshold (0.35 = more sensitive)
threshold = 0.35
y_pred = (y_proba > threshold).astype(int)

# Metrics
auroc = roc_auc_score(y_test, y_proba)
auprc = average_precision_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("📊 Evaluation Metrics:")
print(f"AUROC: {auroc:.4f}")
print(f"AUPRC: {auprc:.4f}\n")
print(report)
print(cm)

# ==========================================
# STEP 6: Save Model
# ==========================================
joblib.dump(model, "sepsis_model.pkl")
print("\n💾 Model saved successfully as 'sepsis_model.pkl'")
