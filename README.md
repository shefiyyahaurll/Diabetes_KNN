# **Laporan Proyek Machine Learning - Shefiyyah Aurellia Wahyudi**<br>

## Prediksi Diabetes menggunakan Algoritma K-Nearest Neighbors (KNN)

## Domain Proyek
Meningkatnya kesulitan dalam mengekstraksi informasi yang berguna untuk mendukung keputusan dari sistem informasi medis yang besar dan kompleks di rumah sakit dan institusi medis modern. Analisis data manual tradisional menjadi tidak efisien, dan ada kebutuhan akan metode analisis berbasis komputer yang efisien untuk mengatasi tantangan ini. Pengenalan pembelajaran mesin ke dalam analisis medis telah terbukti meningkatkan akurasi diagnostik, mengurangi biaya, dan meminimalkan kebutuhan sumber daya manusia. Urgensi untuk mengatasi masalah ini terletak pada kebutuhan akan alat diagnostik yang akurat dan efisien untuk mengidentifikasi dan mengelola diabetes secara tepat waktu. Kompleksitas dan volume data medis dalam sistem perawatan kesehatan modern membuat analisis manual tradisional menjadi tidak efisien, sehingga memerlukan penggunaan metode analisis berbasis komputer yang canggih[1].<br>
Machine Learning telah menunjukkan keefektifannya dalam diagnosa medis, pengurangan biaya, dan meminimalkan kebutuhan sumber daya manusia. Sebagai contoh, sebuah penelitian yang diterbitkan dalam Journal of Basrah Researches menunjukkan bahwa teknik pembelajaran mesin, termasuk algoritma support vector machine (SVM) dan k-nearest neighbor (KNN), mencapai akurasi yang tinggi dalam prediksi diabetes, sehingga meningkatkan kemampuan diagnostik. Selain itu, sebuah artikel ulasan di Korea Journal of Artificial Intelligence menyoroti potensi pembelajaran mesin, khususnya SVM, dalam mengurangi biaya medis dan meminimalkan kebutuhan sumber daya manusia dalam diagnosis diabetes[2].<br>
Untuk memberikan wawasan tentang fitur-fitur penting untuk metode klasifikasi KNN dalam memprediksi diabetes berdasarkan PIMA Indian Database. Dengan mengatasi kesenjangan ini, bertujuan untuk berkontribusi pada pemahaman tentang fitur-fitur penting untuk prediksi diabetes yang akurat[2].<br>
Penggunaan pembelajaran mesin dalam diagnosis diabetes telah memberikan dampak positif yang signifikan, termasuk pengurangan biaya dan peningkatan akurasi. Sebuah studi yang diterbitkan di Sensors pada tahun 2020 menunjukkan bahwa penerapan pembelajaran mesin dalam diagnosis diabetes dapat mengurangi biaya perawatan kesehatan dengan meningkatkan akurasi prediksi diabetes. Selain itu, penelitian yang diterbitkan di Journal of Basrah Researches pada tahun 2021 juga menunjukkan bahwa pembelajaran mesin telah berhasil meningkatkan akurasi prediksi diabetes, yang pada gilirannya dapat mengurangi biaya perawatan kesehatan dan meningkatkan efisiensi diagnosis[2].<br>

## Business Understanding<br>
Analisis data manual tradisional tidak efisien karena memerlukan waktu dan sumber daya manusia yang besar, terutama ketika menghadapi volume data medis yang kompleks dan besar. Proses manual ini rentan terhadap kesalahan manusia dan dapat memakan waktu yang lama. Selain itu, data medis yang kompleks memerlukan pemrosesan yang cermat dan analisis yang mendalam untuk mengidentifikasi pola dan hubungan yang relevan. Oleh karena itu, metode berbasis komputer diperlukan untuk meningkatkan efisiensi dan akurasi analisis data medis.<br>

Metode berbasis komputer, seperti pembelajaran mesin dan algoritma klasifikasi seperti KNN, SVM, dan regresi logistik, dapat mengatasi kompleksitas dan volume data medis dengan lebih efisien. Mereka mampu melakukan analisis cepat dan akurat terhadap data yang besar, mengidentifikasi pola yang kompleks, dan memprediksi hasil dengan tingkat akurasi yang tinggi. Selain itu, metode berbasis komputer juga dapat mengurangi ketergantungan pada sumber daya manusia, sehingga memungkinkan penghematan waktu dan biaya dalam proses analisis data medis.
1. Dalam diagnosis medis, seperti diagnosis diabetes Pima Indian, terletak pada kebutuhan akan alat diagnostik yang akurat dan efisien untuk mengidentifikasi dan mengelola diabetes secara tepat waktu. 
2. Ketika sistem informasi medis di rumah sakit modern dan institusi medis menjadi semakin besar, hal ini menyebabkan kesulitan besar dalam mengekstraksi informasi yang berguna untuk mendukung keputusan. Analisis data manual tradisional menjadi tidak efisien dan metode untuk analisis berbasis komputer yang efisien menjadi sangat penting. Telah terbukti bahwa manfaat memperkenalkan pembelajaran mesin ke dalam analisis medis adalah untuk meningkatkan akurasi diagnostik, mengurangi biaya, dan mengurangi sumber daya manusia.<br>

### Problem Statements
1. Bagaimana penggunaan metode KNN dalam menganalisis fitur-fitur penting dalam database PIMA Indian untuk memprediksi diabetes?
2. Berapa akurasi yang dihasilkan dari model KNN untuk memprediksi diabetes?
3. Apakah akurasi yang dihasilkan merupakan akurasi yang baik dan akurat?

### **Goals**<br>
1. Menganalisis fitur-fitur penting dalam database PIMA Indian menggunakan metode KNN untuk klasifikasi diabetes.<br>
Terbukti pada hasil tahap multivariate analysis terdapat fitur yang memiliki nilai kolerasi tertinggi yaitu 'glucose' dimana memiliki nilai kolerasi 0.5<br>

2. nilai akurasi mse yang dihasilkan oleh model rendah.<br>
Pada tahap evaluasi menghasilkan nilai MSE train yaitu 0.000134 dan MSE test yaitu 0.000141 ini artinya model KNN tampaknya berkinerja baik pada dataset pelatihan dan pengujian, menunjukkan kesalahan yang rendah dalam memprediksi pasien diabetes atau bukan.<br>

3. Akurasi nilai mse yang dihasilkan merupakan yang akurat atau baik memiliki kesalahan rendah.<br>
Pada tahap evaluasi menghasilkan Mean Squared Error (MSE) yang diperoleh yaitu MSE train yaitu 0.000134 dan MSE test yaitu 0.000141 termasuk rendah menunjukkan bahwa model memiliki tingkat kesalahan yang rendah dalam memprediksi nilai sebenarnya. Dalam konteks diagnosis diabetes, implementasi dari nilai MSE yang rendah dapat memberikan keyakinan yang lebih tinggi dalam hasil prediksi model terhadap kondisi diabetes seseorang. Dengan nilai MSE yang rendah, model dapat memberikan prediksi yang lebih akurat dan dapat diandalkan dalam mengidentifikasi kemungkinan adanya diabetes pada pasien.

##### **Solution statements**<br>
Dengan membuat model machine learning dengan Model Development dari K-Nearest Neighbor dapat mengetahui yang mana saja pasien yang terkena penyakit diabetes<br>

## Data Understanding
Berikut link Diabetes dataset dari kaggle https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database<br>
- Terdapat 768 baris (records atau jumlah pengamatan) dalam dataset.
- Terdapat 9 kolom yaitu: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.<br>
- Terdapat 7 kolom numerik dengan tipe data int64 yaitu: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, Age, Outcome. 
- Terdapat 2 kolom numerik dengan tipe data float64, yaitu: BMI, DiabetesPedigreeFunction

### Variabel-variabel pada dataset adalah sebagai berikut:
- Pregnancies: Berapa kali hamil
- Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mm U/ml)
- BMI: Indeks massa tubuh (berat badan dalam kg/(tinggi badan dalam m)^2)
- DiabetesPedigreeFunction: Fungsi silsilah Diabetes
- Age: umur(tahun)
- Outcome: Variabel kelas (0 atau 1) 268 dari 768 adalah 1, yang lainnya adalah 0 <br>

### Interpretasi dan Dampak Potensial dari variabel-variabel pada dataset adalah sebagai berikut:<br>

1. **Pregnancies (Berapa kali hamil):**
   - **Interpretasi:** Menunjukkan jumlah kehamilan yang pernah dialami oleh individu.
   - **Dampak Potensial:** Jumlah kehamilan dapat memberikan indikasi tentang kecenderungan seseorang terhadap diabetes. Beberapa penelitian menunjukkan bahwa multiparitas (banyak kehamilan) dapat berkontribusi pada risiko diabetes gestasional.

2. **Glucose (Konsentrasi glukosa plasma 2 jam):**
   - **Interpretasi:** Menunjukkan konsentrasi glukosa dalam plasma setelah tes toleransi glukosa oral.
   - **Dampak Potensial:** Tingkat glukosa yang tinggi dapat menjadi indikator diabetes atau pre-diabetes. Pengukuran ini membantu menilai seberapa baik tubuh dapat mengelola glukosa.

3. **BloodPressure (Tekanan darah diastolik):**
   - **Interpretasi:** Menunjukkan tekanan darah diastolik, yaitu tekanan pada dinding arteri ketika jantung sedang beristirahat.
   - **Dampak Potensial:** Tekanan darah yang tinggi dapat meningkatkan risiko diabetes. Pemantauan tekanan darah penting karena hubungannya dengan penyakit jantung dan vaskular.

4. **SkinThickness (Ketebalan lipatan kulit trisep):**
   - **Interpretasi:** Mengukur ketebalan lipatan kulit di daerah trisep, yang dapat memberikan perkiraan lemak tubuh.
   - **Dampak Potensial:** Meskipun tidak langsung terkait dengan diabetes, ketebalan lemak tubuh dapat memiliki implikasi terhadap resistensi insulin dan faktor risiko lainnya.

5. **Insulin (Insulin serum 2 jam):**
   - **Interpretasi:** Mengukur konsentrasi insulin dalam serum 2 jam setelah tes toleransi glukosa oral.
   - **Dampak Potensial:** Tingkat insulin yang tinggi dapat mengindikasikan resistensi insulin, yang terkait dengan diabetes tipe 2. Pengukuran ini dapat membantu memahami respons tubuh terhadap glukosa.

6. **BMI (Indeks Massa Tubuh):**
   - **Interpretasi:** Menghitung indeks massa tubuh sebagai indikator obesitas.
   - **Dampak Potensial:** Obesitas adalah faktor risiko utama untuk diabetes. BMI tinggi dapat menunjukkan kelebihan berat badan, yang dapat meningkatkan risiko resistensi insulin.

7. **DiabetesPedigreeFunction (Fungsi Silsilah Diabetes):**
   - **Interpretasi:** Menunjukkan sejauh mana ada riwayat diabetes dalam keluarga.
   - **Dampak Potensial:** Riwayat keluarga dapat menjadi faktor risiko untuk diabetes. Fungsi silsilah ini memberikan bobot pada seberapa besar pengaruh faktor genetik terhadap kemungkinan seseorang mengembangkan diabetes.

8. **Age (Umur):**
   - **Interpretasi:** Menunjukkan usia individu dalam tahun.
   - **Dampak Potensial:** Risiko diabetes meningkat seiring bertambahnya usia. Pemahaman tentang usia dapat membantu dalam menilai risiko secara umum.

Melalui pemahaman mendalam terhadap setiap variabel ini, model machine learning dapat dilatih untuk memprediksi kemungkinan seseorang mengembangkan diabetes berdasarkan karakteristik klinis mereka. Dengan menganalisis variabel-variabel ini, dapat dilakukan identifikasi faktor-faktor risiko dan pencegahan lebih dini.<br>

### multivariate analysis<br>
![Teks alternatif](gambar/output1.png)<br>
Pada  gambar multivariate analysis diatas  ditemukan bahwa Pada baris paling bawah (nilai korelasi terhadap kolom Outcome), terlihat hampir semua kotak cenderung berwarna biru, yang berarti nilainya mendekati 0. Ini menandakan bahwa hampir semua fitur tidak memiliki hubungan signifikan dengan dengan kolom Outcome. Korelasi fitur tertinggi terhadap 'outcome' atau target adalah fitur 'glucose' dimana memiliki nilai kolerasi 0.5<br>

## Data Preparation

- Menangani Missing Value<br>
Dari hasil fungsi describe(), nilai minimum untuk **kolom Glucose, BloodPressure, SkinThickness, Insulin, BMI  adalah 0**. BloodPressure, SkinThickness, Insulin, BMI adalah beberapa istilah yang umumnya terkait dengan masalah kesehatan dan diagnosis, terutama dalam konteks diabetes yang memiliki jumlah dan tidak mungkin 0. Maka dari itu ini merupakan data yang tidak valid atau sering disebut missing value.<br>

  Yang perlu dilakukan diantaranya:<br>
  1. mengecek jumlah 0 di kolom Glucose, BloodPressure, SkinThickness, Insulin, BMI<br>
  2. mengganti angka 0 dengan N/A atau kosong<
  3. mengganti nilai yang kosong dengan nilai rata-rata dari kolom tersebut karena data yang hilang atau kosong banyak, maka dari itu nilai yang kosong dapat diganti dengan nilai rata-rata<br>

- Pembagian dataset dengan fungsi train_test_split dari library sklearn.<br>
Pada tahap ini menggunakan beberapa variabel dan parameter diantaranya:
  - Variabel X menyimpan fitur-fitur dari dataset kecuali kolom "Outcome".
  - Variabel y menyimpan kolom "Outcome", yang berisi label apakah pasien memiliki diabetes atau tidak.
  - Fungsi train_test_split digunakan untuk membagi dataset menjadi data pelatihan dan data uji. Data uji sebesar 10% dari keseluruhan dataset (ditentukan oleh parameter test_size=0.1).
  - Parameter random_state=123 digunakan untuk menetapkan seed agar hasil pemisahan dapat direproduksi dengan cara yang sama setiap kali kode dijalankan.
  
- Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. <br>
Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, ini tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Akan menggunakan teknik StandarScaler dari library Scikitlearn, <br>
StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.<br>
Untuk menghindari kebocoran informasi pada data uji hanya akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi akan melakukan standarisasi pada data uji.


## Modeling
Algoritma yang digunakan yaitu KNN (K-Nearest Neighbors) adalah salah satu algoritma machine learning yang digunakan untuk masalah klasifikasi dan regresi. Algoritma ini bekerja berdasarkan prinsip bahwa suatu data dapat diklasifikasikan atau diestimasi nilai-nilainya berdasarkan mayoritas label data yang paling dekat dengannya dalam ruang fitur.<br>

Algoritma  K-Nearest Neighbors (KNN) untuk tugas regresi digunakan dengan KNN akan menggunakan 10 tetangga terdekat untuk memprediksi nilai regresi suatu observasi, menggunakan nilai n_neighbors=10. Dengan menggunakan nilai n_neighbors=10 untuk mencapai nilai MSE yang rendah<br>

Model KNN cocok untuk projek ini karena KNN merupakan metode klasifikasi yang sederhana dan mudah diimplementasikan. Selain itu, KNN juga cocok untuk dataset PIMA Indian karena mampu menangani data numerik dan kategorikal dengan baik. Selain itu, KNN juga cocok untuk projek ini karena mampu memberikan hasil yang baik dalam menganalisis fitur-fitur penting dalam dataset PIMA Indian untuk memprediksi diabetes

## Evaluation
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.<br>

Metrik yang akan digunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut<br>

$$ MSE = \frac{1}{N}\sum_{i=1}^{N}(y{i}-y{pred{i}})^{2} $$

Keterangan:<br>

- N = jumlah dataset

- yi = nilai sebenarnya

- y_pred = nilai prediksi<br>

![Teks alternatif](gambar/output.png)<br>

Hasil penerapan metrik evaluasi yang digunakan sesuai dengan konteks data, pernyataan masalah, dan solusi yang diinginkan di awal proyek. Dalam konteks data PIMA Indian untuk prediksi diabetes, penelitian menggunakan metrik evaluasi akurasi untuk mengukur sejauh mana model KNN mampu memprediksi diabetes dengan benar. Memiliki nilai MSE train yaitu 0.000134 dan MSE test yaitu 0.000141 ini artinya model KNN tampaknya berkinerja baik pada dataset pelatihan dan pengujian, menunjukkan kesalahan yang rendah dalam memprediksi pasien diabetes atau bukan.<br>

Mean Squared Error (MSE) yang rendah menunjukkan bahwa model memiliki tingkat kesalahan yang rendah dalam memprediksi nilai sebenarnya. Dalam konteks diagnosis diabetes, implementasi dari nilai MSE yang rendah dapat memberikan keyakinan yang lebih tinggi dalam hasil prediksi model terhadap kondisi diabetes seseorang. Dengan nilai MSE yang rendah, model dapat memberikan prediksi yang lebih akurat dan dapat diandalkan dalam mengidentifikasi kemungkinan adanya diabetes pada pasien.<br>

Implementasi strategi untuk mengatasi masalah diagnosis diabetes dapat melibatkan penerapan model dengan nilai MSE yang rendah sebagai alat bantu dalam proses diagnosis. Dengan model yang memiliki tingkat kesalahan rendah, dokter atau tenaga medis dapat menggunakan hasil prediksi model sebagai salah satu pertimbangan dalam menentukan diagnosis diabetes pada pasien. Hal ini dapat membantu dalam mendukung keputusan medis yang lebih tepat dan akurat.


  Referensi:<br>
  [1]	K. Kayaer and T. Yildirim, “Medical Diagnosis on Pima Indian Diabetes Using General Regression Neural Networks,” Iternational Conf. Artif. Neural Networks Neural Inf. Process., no. January 2003, pp. 181–184, 2003, [Online]. Available: www.yildiz.edu.tr/~tulay/publications/Icann-Iconip2003-2.pdf<br>
  [2]	A. Perdana, A. Hermawan, and D. Avianto, “Analyze Important Features of PIMA Indian Database For Diabetes Prediction Using KNN,” J. Sisfokom (Sistem Inf. dan Komputer), vol. 12, no. 1, pp. 70–75, 2023, doi: 10.32736/sisfokom.v12i1.1598.<br>



