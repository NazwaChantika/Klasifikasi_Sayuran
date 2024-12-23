# Sistem Klasifikasi Citra Sayur untuk Mendukung Pola Makan Sehat 👩🏻‍🍳🌽

## Overview Project
Sayuran adalah salah satu komponen utama dalam pola makan sehat karena kaya akan nutrisi, termasuk vitamin, mineral, serat, dan antioksidan. Namun, banyak orang menghadapi tantangan dalam mengenali jenis-jenis sayuran dan memahami manfaat nutrisinya. Proyek ini bertujuan untuk mengembangkan Sistem Klasifikasi Citra Sayur berbasis kecerdasan buatan sebagai alat edukasi untuk mendukung pola makan sehat.

Link Dataset yang digunakan [Dataset_Vegetables](https://www.kaggle.com/code/chitwanmanchanda/vegetable-image-classification-using-cnn?kernelSessionId=84747681)
Preprocessing yang digunakan yaitu Resize, Normalisasi, dan Augmentasi. Dan model yang digunakan yaitu MobileNetV2 dan InceptionV3.

**InceptionV3 Architecture**
![image](assets/inceptionV3.jpg)

**MobileNetV2 Architecture**
![image](assets/mobilenetv2.jpg)

## Overview Dataset
Dataset yang digunakan adalah Klasifikasi Sayuran dengan link sebagai [berikut](https://www.kaggle.com/code/chitwanmanchanda/vegetable-image-classification-using-cnn?kernelSessionId=84747681). Dataset terdiri atas 21000 gambar yang terbagi menjadi 70% sebagai Training Set, 15% sebagai Validation Set, dan 15% sebagai Testing Set, dimana pada setiap Set, terdapat 15 Label Class yaitu Bean, Bitter Ground, Bottle Ground, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.

## Preprocessing & Modelling 

### CNN Model
Preprocessing

Preprocessing yang dilakukan antara lain adalah resizing (150, 150), lalu rescale / normalization dengan rentang 1./255, dilanjut dengan melakukan splitting dataset menjadi 3 (Training, Validation, dan Testing) sesuai dengan penjelasan pada Dataset.

Modelling
Hasil dari CNN Model yang telah dibangun sebagai berikut : 
![image]()

Model Evaluation
Berikut adalah hasil dari fitting CNN Model yang telah dibangun :
![image]()

1[image]()
