import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

image_base64 = get_base64_image("src/static/background/background.jpg")

background_css = f"""
<style>
body {{
    background-image: url('data:image/jpg;base64,{image_base64}');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(background_css, unsafe_allow_html=True)

# Judul aplikasi
st.markdown(    """
    <div style="text-align: center; margin-bottom: 20px; font-size: 40px">
        Klasifikasi Citra Sayuran
    </div>
    """,
    unsafe_allow_html=True
)

# Tambahkan CSS untuk latar belakang

st.markdown(
    """
    <div style="text-align: justify; margin-bottom: 20px;">
            Sayuran adalah bagian penting dari pola makan sehat karena kaya akan vitamin, mineral, serat, dan antioksidan. 
        Konsumsi sayuran secara rutin dapat membantu menjaga kesehatan tubuh dan mencegah berbagai penyakit kronis.
    </div>

    <div style="text-align: justify; margin-bottom: 20px;">
            Aplikasi ini dibuat untuk membantu mengidentifikasi jenis sayuran dengan menggunakan teknologi pengolahan citra dan pembelajaran mesin. 
        Silakan unggah gambar sayuran, pilih model yang tersedia, dan tekan tombol Predict untuk mendapatkan hasil prediksi.
    </div>
    """,
    unsafe_allow_html=True
)

# Fungsi prediksi
def predict(uploaded_image, model_path):
    # Daftar kelas
    class_names = [
        "Bean",
        "Bitter_Gourd",
        "Bottle_Gourd",
        "Brinjal",
        "Broccoli",
        "Cabbage",
        "Capsicum",
        "Carrot",
        "Cauliflower",
        "Cucumber",
        "Papaya",
        "Potato",
        "Pumpkin",
        "Radish",
        "Tomato",
    ]

    class_descriptions = [
        "Kacang hijau mengandung protein nabati tinggi, serat, serta mineral seperti magnesium dan zat besi yang baik untuk kesehatan jantung dan pencernaan.",
        "Pare mengandung senyawa yang dapat membantu menurunkan kadar gula darah, serta kaya akan vitamin C dan serat.",
        "Labu botol kaya akan air dan serat, baik untuk menjaga hidrasi tubuh dan mendukung kesehatan pencernaan.",
        "Terong mengandung antioksidan seperti nasunin yang baik untuk kesehatan otak, serta kaya akan serat dan vitamin B.",
        "Brokoli mengandung vitamin C, K, dan serat tinggi, serta senyawa sulforaphane yang dapat membantu melawan kanker.",
        "Kubis kaya akan vitamin C dan K, serta memiliki senyawa sulfur yang bermanfaat untuk detoksifikasi tubuh.",
        "Paprika mengandung vitamin A, C, dan E, serta antioksidan yang membantu meningkatkan kekebalan tubuh dan kesehatan kulit.",
        "Wortel kaya akan beta-karoten, yang diubah menjadi vitamin A di tubuh, penting untuk kesehatan mata dan kulit.",
        "Kembang kol adalah sumber vitamin C dan K, serta mengandung antioksidan yang baik untuk kesehatan tubuh secara keseluruhan.",
        "Mentimun mengandung air yang tinggi, baik untuk hidrasi, serta memiliki vitamin K dan potasium untuk kesehatan tulang.",
        "Pepaya mengandung enzim papain, vitamin C, dan serat yang baik untuk pencernaan dan kesehatan kulit.",
        "Kentang adalah sumber karbohidrat kompleks yang menyediakan energi, serta mengandung vitamin C dan potasium.",
        "Labu kuning kaya akan beta-karoten, vitamin A, dan serat, baik untuk kesehatan mata dan pencernaan.",
        "Lobak mengandung vitamin C, folat, dan serat tinggi, membantu dalam detoksifikasi tubuh dan kesehatan kulit.",
        "Tomat mengandung likopen, vitamin C, dan potasium yang baik untuk kesehatan jantung dan kulit."
    ]


    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Muat model
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])  # Hitung probabilitas
    return class_names[np.argmax(score)], class_descriptions[np.argmax(score)], 100 * np.max(score)  # Prediksi label dan confidence

# Pilihan model
model_option = st.selectbox("Pilih model untuk prediksi:", ("InceptionV3", "MobileNetV2"))

# Tentukan path model berdasarkan pilihan
if model_option == "InceptionV3":
    model_path = Path(__file__).parent / "Model/Image/InceptionV3/model.h5"
else:
    model_path = Path(__file__).parent / "Model/Image/MobileNetV2/model.h5"

# Komponen file uploader untuk banyak file
uploads = st.file_uploader("Unggah citra untuk mendapatkan hasil prediksi", type=["png", "jpg"], accept_multiple_files=True)

# Tombol prediksi
if st.button("Predict", type="primary"):
    if uploads:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_container_width=True)

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, label_description, confidence = predict(upload, model_path)
                    st.write(f"Image: **{upload.name}**")
                    st.write(f"Label : **{label}**")
                    st.write(f"Confidence: **{confidence:.5f}%**")
                    st.write(f"Keterangan Sayuran: **{label_description}**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
