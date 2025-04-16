# 🧪 UTS Pemodelan & Pemrograman Klasifikasi  
## 🧠 Klasifikasi Kualitas Anggur menggunakan Random Forest

---

## 1. Deskripsi Umum

Proyek ini bertujuan untuk mengklasifikasikan kualitas anggur ke dalam dua kelas: **rendah (0)** dan **tinggi (1)**. Model yang digunakan adalah **Random Forest Classifier** dengan preprocessing berupa **normalisasi dan stratifikasi split**.

---

## 2. Langkah-Langkah Pemodelan

### 📥 Import Library
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

### 📊 Load dan Eksplorasi Dataset
```python
df_train = pd.read_csv('data_training.csv')
df_test = pd.read_csv('data_testing.csv')
df_train.head(), df_train.info()
df_train.describe()
```

### ❓ Cek Missing Value
```python
df_train.isnull().sum()
df_test.isnull().sum()
```

### 📈 Visualisasi Data
```python
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df_train, palette='viridis')
plt.title('Distribusi Nilai Kualitas Anggur')
plt.xlabel('Kualitas')
plt.ylabel('Jumlah Sampel')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show
```

### 🔁 Binerisasi Target
```python
df_train_binary = df_train.copy()
df_test_binary = df_test.copy()
df_train_binary['quality'] = (df_train_binary['quality'] >= 6).astype(int)
df_train_binary['quality'].value_counts()
```

### 🧹 Preprocessing
```python
X = df_train_binary.drop(columns=['quality', 'Id'])
y = df_train_binary['quality']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

### 🌲 Pelatihan Model Random Forest
```python
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
```

### 📊 Evaluasi Model
```python
y_pred = rf.predict(X_val_scaled)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred, output_dict=True)
print(classification_report(y_val, y_pred))
```

### 🔍 Prediksi Data Uji
```python
X_test = df_test.drop(columns=['Id'])
X_test_scaled = scaler.transform(X_test)
test_predictions = rf.predict(X_test_scaled)
```

### 💾 Simpan Output
```python
df_test_results = df_test.copy()
df_test_results['predicted_quality'] = test_predictions
df_test_results[['Id','predicted_quality']].to_csv('hasilprediksi_013.csv', index=False)
```

---

## 3. Kesimpulan

Model **Random Forest** mampu mengklasifikasikan kualitas anggur dengan cukup baik setelah preprocessing dilakukan. Proses binarisasi membantu menyederhanakan masalah klasifikasi dan menghasilkan distribusi label yang lebih seimbang. Evaluasi menunjukkan performa yang memadai dan model berhasil diaplikasikan pada data test.

---
