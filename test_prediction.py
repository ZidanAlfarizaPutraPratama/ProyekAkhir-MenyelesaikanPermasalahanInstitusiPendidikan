import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load('model/dropout_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Urutan fitur yang dipakai model/scaler
features = list(scaler.feature_names_in_)

# Tampilkan urutan kelas asli yang dipakai model
print("Urutan kelas model:", model.classes_)

# Mapping label ke nama kelas, sesuaikan dengan urutan model.classes_
# Misal: kalau model.classes_ = [0,1,2] dan 0=Graduate,1=Dropout,2=Enrolled
status_map = {
    model.classes_[0]: "Graduate",
    model.classes_[1]: "Dropout",
    model.classes_[2]: "Enrolled"
}

sample_datas = [
    # Graduate (nilai tinggi)
    {
        "Marital_status": 0, "Application_mode": 1, "Application_order": 1, "Course": 10,
        "Daytime_evening_attendance": 0, "Previous_qualification": 15, "Previous_qualification_grade": 16.5,
        "Nacionality": 1, "Mothers_qualification": 12, "Fathers_qualification": 12, "Mothers_occupation": 5,
        "Fathers_occupation": 5, "Admission_grade": 18.0, "Displaced": 0, "Educational_special_needs": 0,
        "Debtor": 0, "Tuition_fees_up_to_date": 1, "Gender": 0, "Scholarship_holder": 1, "Age_at_enrollment": 20,
        "International": 0, "Curricular_units_1st_sem_credited": 60, "Curricular_units_1st_sem_enrolled": 60,
        "Curricular_units_1st_sem_evaluations": 60, "Curricular_units_1st_sem_approved": 60,
        "Curricular_units_1st_sem_grade": 17.0, "Curricular_units_1st_sem_without_evaluations": 0,
        "Curricular_units_2nd_sem_credited": 60, "Curricular_units_2nd_sem_enrolled": 60,
        "Curricular_units_2nd_sem_evaluations": 60, "Curricular_units_2nd_sem_approved": 60,
        "Curricular_units_2nd_sem_grade": 17.0, "Curricular_units_2nd_sem_without_evaluations": 0,
        "Unemployment_rate": 5.0, "Inflation_rate": 2.0, "GDP": 15000.0
    },
    # Dropout (nilai rendah)
    {
        "Marital_status": 1, "Application_mode": 2, "Application_order": 3, "Course": 5,
        "Daytime_evening_attendance": 1, "Previous_qualification": 10, "Previous_qualification_grade": 8.0,
        "Nacionality": 1, "Mothers_qualification": 5, "Fathers_qualification": 5, "Mothers_occupation": 3,
        "Fathers_occupation": 3, "Admission_grade": 10.0, "Displaced": 1, "Educational_special_needs": 1,
        "Debtor": 1, "Tuition_fees_up_to_date": 0, "Gender": 1, "Scholarship_holder": 0, "Age_at_enrollment": 23,
        "International": 1, "Curricular_units_1st_sem_credited": 10, "Curricular_units_1st_sem_enrolled": 12,
        "Curricular_units_1st_sem_evaluations": 8, "Curricular_units_1st_sem_approved": 5,
        "Curricular_units_1st_sem_grade": 7.0, "Curricular_units_1st_sem_without_evaluations": 2,
        "Curricular_units_2nd_sem_credited": 5, "Curricular_units_2nd_sem_enrolled": 8,
        "Curricular_units_2nd_sem_evaluations": 4, "Curricular_units_2nd_sem_approved": 3,
        "Curricular_units_2nd_sem_grade": 6.0, "Curricular_units_2nd_sem_without_evaluations": 3,
        "Unemployment_rate": 12.0, "Inflation_rate": 7.0, "GDP": 8000.0
    },
    # Enrolled (nilai sedang)
    {
        "Marital_status": 0, "Application_mode": 1, "Application_order": 2, "Course": 7,
        "Daytime_evening_attendance": 0, "Previous_qualification": 12, "Previous_qualification_grade": 12.0,
        "Nacionality": 1, "Mothers_qualification": 10, "Fathers_qualification": 10, "Mothers_occupation": 4,
        "Fathers_occupation": 4, "Admission_grade": 14.0, "Displaced": 0, "Educational_special_needs": 0,
        "Debtor": 0, "Tuition_fees_up_to_date": 1, "Gender": 0, "Scholarship_holder": 0, "Age_at_enrollment": 21,
        "International": 0, "Curricular_units_1st_sem_credited": 30, "Curricular_units_1st_sem_enrolled": 30,
        "Curricular_units_1st_sem_evaluations": 30, "Curricular_units_1st_sem_approved": 25,
        "Curricular_units_1st_sem_grade": 11.0, "Curricular_units_1st_sem_without_evaluations": 5,
        "Curricular_units_2nd_sem_credited": 30, "Curricular_units_2nd_sem_enrolled": 30,
        "Curricular_units_2nd_sem_evaluations": 30, "Curricular_units_2nd_sem_approved": 20,
        "Curricular_units_2nd_sem_grade": 10.0, "Curricular_units_2nd_sem_without_evaluations": 5,
        "Unemployment_rate": 7.0, "Inflation_rate": 4.0, "GDP": 12000.0
    }
]

for idx, data in enumerate(sample_datas):
    df = pd.DataFrame([data], columns=features)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    proba = model.predict_proba(df_scaled)

    print(f"\nContoh data ke-{idx+1}:")
    print(df.T)
    print(f"Prediksi label kelas (angka): {pred[0]}")
    print(f"Prediksi model: {status_map.get(pred[0], 'Unknown')}")
    print("Probabilitas kelas:")
    for i, p in enumerate(proba[0]):
        print(f"  {status_map.get(model.classes_[i], 'Unknown')}: {p:.2%}")
