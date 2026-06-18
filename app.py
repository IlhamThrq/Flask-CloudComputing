from flask import Flask, render_template, request
import pickle
import pandas as pd
import warnings

# Mengabaikan warning versi scikit-learn agar terminal lebih bersih (opsional)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Memuat model (yang ternyata berisi list of models) dan scaler [cite: 76, 77]
with open('model.pkl', 'rb') as f:
    models_list = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Sesuaikan daftar ini dengan model yang ada di dalam file .pkl milikmu
# Berdasarkan log error, kamu sepertinya me-load Decision Tree dan SVC
model_names = ["Decision Tree", "SVC"] 

@app.route("/")
def index():
    # Menampilkan halaman utama (form input) [cite: 79]
    return render_template('index.html', model_names=model_names)

@app.route("/predict", methods=['POST'])
def predict():
    # Menangkap data yang dikirim dari form HTML [cite: 81, 82, 83, 84, 86, 87, 88, 89]
    data = {
        'Pregnancies': int(request.form['Pregnancies']),
        'Glucose': int(request.form['Glucose']),
        'BloodPressure': int(request.form['BloodPressure']),
        'SkinThickness': int(request.form['SkinThickness']),
        'Insulin': int(request.form['Insulin']),
        'BMI': float(request.form['BMI']),
        'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
        'Age': int(request.form['Age'])
    }
    
    # Mengubah dictionary data menjadi DataFrame Pandas [cite: 90]
    df = pd.DataFrame([data])
    
    # Melakukan standardisasi/scaling data menggunakan scaler [cite: 91]
    df_scaled = scaler.transform(df)
    
    # MENYELESAIKAN ERROR: 
    # 1. Ambil nama model yang dipilih user dari dropdown HTML
    selected_model_name = request.form['model']
    
    # 2. Cari urutan (index) model tersebut di variabel model_names
    # 3. Ekstrak model yang sesuai dari models_list [cite: 92]
    model_index = model_names.index(selected_model_name)
    clf = models_list[model_index]
    
    # 4. Melakukan prediksi dengan model yang sudah diekstrak [cite: 93]
    pred = clf.predict(df_scaled)[0]
    
    # Menentukan hasil prediksi [cite: 103, 104, 328]
    if pred == 1:
        prediction = "Diabetic"
    else:
        prediction = "Non-Diabetic"
        
    # Mengembalikan hasil prediksi ke halaman web [cite: 105, 106]
    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    app.run(debug=True)