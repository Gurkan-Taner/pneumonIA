import streamlit as st
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pickle import load

knn = load(open('knn.pkl', 'rb'))
logistic_regression = load(open('logistic_regression.pkl', 'rb'))
svm = load(open('svm.pkl', 'rb'))


def load_image(image_file, model):
    image = Image.open(image_file).convert('L')
    if(model == 'SVM'):
        image = image.resize((64, 64))
    else:
        image = image.resize((32, 32))
    image_array = (np.array(image) / 255.0).flatten()
    return image_array

def predict_pneumonia(image_array, model):
    prediction = None
    if(model == 'KNN'):
        prediction = knn.predict([image_array])
    elif(model == 'Logistic Regression'):
        prediction = logistic_regression.predict([image_array])
    elif(model == 'SVM'):
        prediction = svm.predict([image_array])
    return prediction[0]

st.title('Pneumonia Detector')
model_choice = st.radio(
    "Sélectionner le modèle à utiliser",
    ["KNN", "Logistic Regression", "SVM"],
    index=None,
)

if(model_choice != None):
    st.write("Veuillez sélectionner ou déposer une image d'un patient pour prédire s'il est atteint de pneumonie.")
    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image sélectionnée.', use_column_width=True)

        image_array = load_image(uploaded_file, model_choice)

        result = predict_pneumonia(image_array, model_choice)
        st.write(f'Prediction: {"Pneumonie" if result == "PNEUMONIA"  else "Normal"}')