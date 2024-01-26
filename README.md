# **Laporan Proyek Machine Learning - Shefiyyah Aurellia Wahyudi**<br>

## Domain Proyek
Banyaknya orang-orang yang belum mengetahui faktor-faktor apa saja yang menyebabkan terkena diabetes. <br>
Indonesia menjadi negara ke-5 dengan prevalensi penyakit diabetes tertinggi di dunia pada 2021 menurut International Diabetes Federation (IDF).

## Business Understanding
1. Deteksi Dini dan Manajemen Penyakit
2. Pengembangan Model Prediktif sebagai Alat Bantu Profesional Kesehatan<br>

Bagian laporan ini mencakup:

### Problem Statements
1. Bagaimana dapat dibangun sebuah model machine learning yang efektif untuk memprediksi apakah seorang pasien dalam dataset memiliki penyakit diabetes atau tidak berdasarkan variabel-variabel yang terdapat dalam dataset?
2. Bagaimana pengaruh masing-masing variabel terhadap keberadaan penyakit diabetes, dan sejauh mana variabel-variabel tersebut dapat menjadi prediktor yang andal?

### **Goals**<br>
1. Dengan Menggunakan Dataset pasien ini, dapat membuat model machine learning untuk memprediksi pakah pasien di dataset punya penyakit diabetes atau bukan
2. Mengetahui hubungan antara penyakit diabetes dengan Glucose, BloodPressure, SkinThickness, Insulin, BMI

##### **Solution statements**<br>
Dengan membuat model machine learning dengan Model Development dari K-Nearest Neighbor dapat mengetahui yang mana saja pasien yang terkena penyakit diabetes<br>

## Data Understanding
Berikut link Diabetes dataset dari kaggle https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database<br>
- Terdapat 768 baris (records atau jumlah pengamatan) dalam dataset.
- Terdapat 9 kolom yaitu: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.

### Variabel-variabel pada dataset adalah sebagai berikut:
- Pregnancies: Berapa kali hamil
- Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mm U/ml)
- BMI: Indeks massa tubuh (berat badan dalam kg/(tinggi badan dalam m)^2)
- DiabetesPedigreeFunction: Fungsi silsilah Diabetes
- Age: umur(tahun)
- Outcome: Variabel kelas (0 atau 1) 268 dari 768 adalah 1, yang lainnya adalah 0

**Rubrik/Kriteria Tambahan**:
menghitung kolerasi antar feature seperti Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, dan Outcome

## Data Preparation
Pada bagian ini akan melakukan 2 tahap persiapan data, yaitu:<br>

- Pembagian dataset dengan fungsi train_test_split dari library sklearn.<br>
Pada tahap ini menggunakan beberapa variabel dan parameter diantaranya:
  - Variabel X menyimpan fitur-fitur dari dataset kecuali kolom "Outcome".
  - Variabel y menyimpan kolom "Outcome", yang berisi label apakah pasien memiliki diabetes atau tidak.
  - Fungsi train_test_split digunakan untuk membagi dataset menjadi data pelatihan dan data uji. Data uji sebesar 10% dari keseluruhan dataset (ditentukan oleh parameter test_size=0.1).
  - Parameter random_state=123 digunakan untuk menetapkan seed agar hasil pemisahan dapat direproduksi dengan cara yang sama setiap kali kode dijalankan.
  
- Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. <br>
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, <br>
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.<br>
Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji. Untuk lebih jelasnya, mari kita terapkan StandardScaler pada data. 


## Modeling
Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi. Namun, dapat mencoba beberapa nilai k yang berbeda, misal: nilai dari 1 hingga 20, kemudian membandingkan mana nilai yang paling sesuai untuk model. <br>

Selanjutnya, untuk menentukan titik mana dalam data yang paling mirip dengan input baru, KNN menggunakan perhitungan ukuran jarak. Metrik ukuran jarak yang digunakan secara default pada library sklearn adalah Minkowski distance. Beberapa metrik ukuran jarak yang juga sering dipakai antara lain: Euclidean distance dan Manhattan distance. Sebagai contoh, jarak Euclidean dihitung sebagai akar kuadrat dari jumlah selisih kuadrat antara titik a dan titik b. Dirumuskan sebagai berikut:<br>

$$ d(x,y) = \sqrt{\sum_{i=1}^{n}(x{i}-y{i})^{2}}  $$

Sedangkan, Minkowski distance merupakan generalisasi dari Euclidean dan Manhattan distance. Untuk menghitungnya, perhatikan rumus berikut:<br>

$$ d(x,y) = (\sum_{i=1}^{n}\left |x{i} -y{i} \right |^{p})\tfrac{1}{p} $$

Model Development yang akan kita buat model machine learning dangan algoritma berikut:<br>

- K-Nearest Neighbor (KNN)<br>

  - KNeighborsRegressor(n_neighbors=10): Membuat objek model K-Nearest Neighbors dengan menentukan jumlah tetangga terdekat sebanyak 10 (ditentukan oleh parameter n_neighbors).
  - knn.fit(X_train, y_train): Melatih model KNN dengan menggunakan data pelatihan (X_train sebagai fitur dan y_train sebagai label).
  - mean_squared_error(y_pred=knn.predict(X_train), y_true=y_train): Mengukur kesalahan model pada data pelatihan dengan menghitung rata-rata dari kuadrat selisih antara nilai prediksi (knn.predict(X_train)) dan nilai sebenarnya (y_train). Hasil dari pengukuran    kesalahan ini kemudian dimasukkan ke dalam suatu lokasi (baris "train_mse" dan kolom "knn") pada suatu dataframe atau struktur data yang disebut models. Ini berguna untuk menyimpan dan memantau kinerja model pada tahap pelatihan.

## Evaluation
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.<br>

Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut<br>

$$ MSE = \frac{1}{N}\sum_{i=1}^{N}(y{i}-y{pred{i}})^{2} $$

Keterangan:<br>

- N = jumlah dataset

- yi = nilai sebenarnya

- y_pred = nilai prediksi<br>

**Pada model KNN yang telah kita buat memiliki nilai MSE train yaitu 0.000134 dan MSE test yaitu 0.000141 ini artinya model KNN Anda tampaknya berkinerja baik pada dataset pelatihan dan pengujian, menunjukkan kesalahan yang rendah dalam memprediksi variabel target.**





