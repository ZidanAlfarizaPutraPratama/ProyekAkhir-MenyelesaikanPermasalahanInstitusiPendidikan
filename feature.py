import joblib

# Load model dan scaler
model = joblib.load('model/dropout_model.joblib')
scaler = joblib.load('model/scaler.joblib')

# Cek fitur pada scaler
if hasattr(scaler, 'feature_names_in_'):
    print("Fitur yang digunakan oleh scaler:")
    print(list(scaler.feature_names_in_))
else:
    print("Scaler tidak memiliki atribut 'feature_names_in_'")

# Cek fitur pada model (bisa error jika model tidak menyimpan nama fitur)
try:
    fitur_model = model.feature_names_in_
    print("Fitur yang digunakan oleh model:")
    print(list(fitur_model))
except AttributeError:
    print("Model tidak memiliki atribut 'feature_names_in_'")
except Exception as e:
    print(f"Terjadi error saat mengambil fitur model: {e}")
