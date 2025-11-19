import os

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# Load model & dataset reference
model = joblib.load("stroke_rf_model.pkl")
df_ref = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preprocessing reference (same as training)
df_ref["bmi"] = df_ref["bmi"].fillna(df_ref["bmi"].median())
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Create label encoders
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ref[col] = df_ref[col].astype(str)
    le.fit(df_ref[col])
    encoders[col] = le

# Kolom fitur mengikuti urutan pelatihan (id tetap disertakan sebagai placeholder)
feature_cols = [c for c in df_ref.columns if c != 'stroke']


def prepare_dataset_for_eval():
    X = df_ref.copy()
    for col in categorical_cols:
        X[col] = encoders[col].transform(X[col].astype(str))
    X = X[feature_cols]
    y = df_ref["stroke"]
    return X, y


def preprocess_input(data):
    X = pd.DataFrame([data])

    # Fill missing BMI
    X["bmi"] = X["bmi"].fillna(df_ref["bmi"].median())

    # Encode categorical
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = encoders[col]

        # Handle unseen label
        if X[col].iloc[0] not in le.classes_:
            le.classes_ = np.append(le.classes_, X[col].iloc[0])

        X[col] = le.transform(X[col])

    # Ensure column order matches training
    X = X[feature_cols]
    return X


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/insight")
def insight():
    gender_counts = df_ref["gender"].value_counts().to_dict()
    stroke_counts = (
        df_ref["stroke"]
        .value_counts()
        .rename({0: "Tidak Stroke", 1: "Stroke"})
        .to_dict()
    )

    age_bins = pd.cut(
        df_ref["age"],
        bins=[0, 20, 40, 60, 80, 120],
        labels=["0-19", "20-39", "40-59", "60-79", "80+"],
        include_lowest=True,
    )
    stroke_rate_by_age = (
        df_ref.groupby(age_bins)["stroke"].mean().fillna(0) * 100
    ).to_dict()

    feature_importance = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(6)
        .to_dict()
    )

    X_all, y_all = prepare_dataset_for_eval()
    y_pred_all = model.predict(X_all)
    cm = confusion_matrix(y_all, y_pred_all, labels=[0, 1])
    cm_dict = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }
    return render_template(
        "insight.html",
        gender_counts=gender_counts,
        stroke_counts=stroke_counts,
        stroke_rate_by_age=stroke_rate_by_age,
        feature_importance=feature_importance,
        confusion_matrix=cm_dict,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Ambil berat badan dan tinggi badan dari form
    weight = float(request.form["weight"])  # dalam kg
    height = float(request.form["height"])  # dalam cm
    
    # Hitung BMI: BMI = berat (kg) / (tinggi (m))²
    height_m = height / 100  # konversi cm ke meter
    bmi = weight / (height_m ** 2)
    
    data = {
        "id": 0,  # placeholder agar sesuai dengan fitur pelatihan
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "ever_married": request.form["ever_married"],
        "work_type": request.form["work_type"],
        "Residence_type": request.form["Residence_type"],
        "avg_glucose_level": float(request.form["avg_glucose_level"]),
        "bmi": bmi,  # BMI dihitung dari berat dan tinggi badan
        "smoking_status": request.form["smoking_status"],
    }

    # Untuk display, tambahkan berat dan tinggi badan yang diinput user
    data_display = {k: v for k, v in data.items() if k != "id"}
    data_display["weight"] = weight
    data_display["height"] = height
    data_display["bmi_calculated"] = round(bmi, 1)  # BMI yang dihitung untuk ditampilkan

    X_input = preprocess_input(data)
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    probability_percent = proba * 100

    if proba < 0.2:
        risk_label = "Risiko Rendah"
    elif proba < 0.5:
        risk_label = "Risiko Sedang"
    else:
        risk_label = "Risiko Tinggi"

    risk_summary = (
        f"Model memperkirakan kemungkinan stroke sebesar {probability_percent:.0f}% "
        f"berdasarkan kombinasi faktor klinis yang Anda masukkan."
    )

    factors = []
    if data["age"] >= 60:
        factors.append("Usia di atas 60 tahun termasuk faktor dengan kontribusi besar.")
    elif data["age"] >= 45:
        factors.append("Usia paruh baya (≥45 tahun) menunjukkan perlunya pemantauan lebih sering.")

    if data["hypertension"] == 1:
        factors.append("Riwayat hipertensi meningkatkan risiko tekanan pembuluh darah.")
    if data["heart_disease"] == 1:
        factors.append("Adanya penyakit jantung menambah beban kerja sistem kardiovaskular.")
    if data["avg_glucose_level"] >= 140:
        factors.append("Kadar glukosa rata-rata tinggi (≥140 mg/dL) dapat memicu kerusakan pembuluh.")
    if data["bmi"] >= 30:
        factors.append("BMI di atas 30 menandakan obesitas yang berkaitan dengan risiko stroke.")
    if data["smoking_status"] in ["smokes", "formerly smoked"]:
        factors.append("Status merokok meningkatkan kemungkinan penumpukan plak.")

    if not factors:
        factors.append("Tidak terdeteksi faktor risiko tinggi pada input, namun tetap perhatikan gaya hidup sehat.")

    return render_template(
        "result.html",
        result=risk_label,
        probability=f"{proba:.2f}",
        probability_percent=f"{probability_percent:.0f}",
        risk_summary=risk_summary,
        factors=factors,
        data=data_display
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
