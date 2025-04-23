import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model/dropout_model.joblib')
scaler = joblib.load('model/scaler.joblib')
features = list(scaler.feature_names_in_)

status_map = {
    model.classes_[0]: "Graduate",
    model.classes_[1]: "Dropout",
    model.classes_[2]: "Enrolled"
}

st.set_page_config(page_title="Prediksi Dropout", layout="wide")
st.title("🎓 Prototype Prediksi Status Siswa")
st.markdown("Isi form berikut untuk memprediksi apakah siswa akan Dropout, Graduate, atau masih Enrolled.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        marital_status = st.selectbox("Marital Status", [0, 1])
        application_mode = st.number_input("Application Mode", 1, 20, 1)
        application_order = st.slider("Application Order", 1, 10, 1)
        course = st.number_input("Course", 1, 20, 10)
        daytime_evening = st.selectbox("Daytime/Evening Attendance", [0, 1])
        prev_qual = st.number_input("Previous Qualification", 1, 20, 12)
        prev_qual_grade = st.slider("Previous Qualification Grade", 0.0, 20.0, 12.0)
        nationality = st.number_input("Nationality", 1, 20, 1)
        mothers_qual = st.number_input("Mother's Qualification", 1, 20, 12)
        fathers_qual = st.number_input("Father's Qualification", 1, 20, 12)
        mothers_occ = st.number_input("Mother's Occupation", 1, 20, 5)
        fathers_occ = st.number_input("Father's Occupation", 1, 20, 5)
        admission_grade = st.slider("Admission Grade", 0.0, 20.0, 15.0)

    with col2:
        displaced = st.selectbox("Displaced", [0, 1])
        special_needs = st.selectbox("Educational Special Needs", [0, 1])
        debtor = st.selectbox("Debtor", [0, 1])
        tuition_up_to_date = st.selectbox("Tuition Fees Up-to-date", [0, 1])
        gender = st.selectbox("Gender", [0, 1])
        scholarship = st.selectbox("Scholarship Holder", [0, 1])
        age = st.number_input("Age at Enrollment", 15, 60, 20)
        international = st.selectbox("International", [0, 1])
        cu1_credited = st.slider("1st Sem - Credited", 0, 60, 30, key="cu1_credited")
        cu1_enrolled = st.slider("1st Sem - Enrolled", 0, 60, 30, key="cu1_enrolled")
        cu1_eval = st.slider("1st Sem - Evaluations", 0, 60, 30, key="cu1_eval")
        cu1_approved = st.slider("1st Sem - Approved", 0, 60, 25, key="cu1_approved")
        cu1_grade = st.slider("1st Sem - Grade", 0.0, 20.0, 12.0, key="cu1_grade")
        cu1_wo_eval = st.slider("1st Sem - Without Evaluations", 0, 60, 5, key="cu1_wo_eval")

    st.markdown("---")
    st.subheader("Semester 2")
    col3, col4 = st.columns(2)

    with col3:
        cu2_credited = st.slider("2nd Sem - Credited", 0, 60, 30, key="cu2_credited")
        cu2_enrolled = st.slider("2nd Sem - Enrolled", 0, 60, 30, key="cu2_enrolled")
        cu2_eval = st.slider("2nd Sem - Evaluations", 0, 60, 30, key="cu2_eval")
        cu2_approved = st.slider("2nd Sem - Approved", 0, 60, 20, key="cu2_approved")
        cu2_grade = st.slider("2nd Sem - Grade", 0.0, 20.0, 10.0, key="cu2_grade")
        cu2_wo_eval = st.slider("2nd Sem - Without Evaluations", 0, 60, 5, key="cu2_wo_eval")

    with col4:
        unemployment_rate = st.slider("Unemployment Rate", 0.0, 20.0, 7.0)
        inflation_rate = st.slider("Inflation Rate", 0.0, 20.0, 3.0)
        gdp = st.number_input("GDP", 0.0, 100000.0, 12000.0)

    submitted = st.form_submit_button("🔍 Prediksi Sekarang")

if submitted:
    # Buat dictionary dari input
    data = {
        "Marital_status": marital_status,
        "Application_mode": application_mode,
        "Application_order": application_order,
        "Course": course,
        "Daytime_evening_attendance": daytime_evening,
        "Previous_qualification": prev_qual,
        "Previous_qualification_grade": prev_qual_grade,
        "Nacionality": nationality,
        "Mothers_qualification": mothers_qual,
        "Fathers_qualification": fathers_qual,
        "Mothers_occupation": mothers_occ,
        "Fathers_occupation": fathers_occ,
        "Admission_grade": admission_grade,
        "Displaced": displaced,
        "Educational_special_needs": special_needs,
        "Debtor": debtor,
        "Tuition_fees_up_to_date": tuition_up_to_date,
        "Gender": gender,
        "Scholarship_holder": scholarship,
        "Age_at_enrollment": age,
        "International": international,
        "Curricular_units_1st_sem_credited": cu1_credited,
        "Curricular_units_1st_sem_enrolled": cu1_enrolled,
        "Curricular_units_1st_sem_evaluations": cu1_eval,
        "Curricular_units_1st_sem_approved": cu1_approved,
        "Curricular_units_1st_sem_grade": cu1_grade,
        "Curricular_units_1st_sem_without_evaluations": cu1_wo_eval,
        "Curricular_units_2nd_sem_credited": cu2_credited,
        "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
        "Curricular_units_2nd_sem_evaluations": cu2_eval,
        "Curricular_units_2nd_sem_approved": cu2_approved,
        "Curricular_units_2nd_sem_grade": cu2_grade,
        "Curricular_units_2nd_sem_without_evaluations": cu2_wo_eval,
        "Unemployment_rate": unemployment_rate,
        "Inflation_rate": inflation_rate,
        "GDP": gdp
    }

    # Prediksi
    df = pd.DataFrame([data], columns=features)
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]

    st.success(f"🎯 Prediksi Status: **{status_map[pred]}**")

    st.markdown("### 🔍 Probabilitas:")
    for i, p in enumerate(proba):
        st.write(f"🟢 {status_map[model.classes_[i]]}: {p:.2%}")
