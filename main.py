import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from PIL import Image


def load_model_and_encoders():
    model = load_model("fashion_model.h5")
    label_encoders = {
        "gender": joblib.load("gender_label_encoder.pkl"),
        "baseColour": joblib.load("baseColour_label_encoder.pkl"),
        "subCategory": joblib.load("subCategory_label_encoder.pkl"),
        "season": joblib.load("season_label_encoder.pkl"),
    }
    return model, label_encoders


def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_attributes(model, label_encoders, img_array):
    predictions = model.predict(img_array)
    gender = label_encoders["gender"].inverse_transform([np.argmax(predictions[0])])[0]
    base_colour = label_encoders["baseColour"].inverse_transform(
        [np.argmax(predictions[1])]
    )[0]
    sub_category = label_encoders["subCategory"].inverse_transform(
        [np.argmax(predictions[2])]
    )[0]
    season = label_encoders["season"].inverse_transform([np.argmax(predictions[3])])[0]
    return gender, base_colour, sub_category, season


st.title("Fashion Attributes Prediction")
st.write("Upload an image of a fashion item to predict its attributes!")

model, label_encoders = load_model_and_encoders()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image"):
        img_array = preprocess_image(uploaded_file)
        gender, base_colour, sub_category, season = predict_attributes(
            model, label_encoders, img_array
        )
    st.success("Prediction Complete!")
    st.write(f"**Predicted Gender:** {gender}")
    st.write(f"**Predicted Base Colour:** {base_colour}")
    st.write(f"**Predicted SubCategory:** {sub_category}")
    st.write(f"**Predicted Season:** {season}")
