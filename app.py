import streamlit as st
import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image, ImageOps
import numpy as np
import cv2

# Download the model file from Google Drive
def download_model():
    gdd.download_file_from_google_drive(file_id='1sikTzO_Imd66SNZvVYe797EBYtDq7nse',
                                        dest_path='./my_model2.hdf5',
                                        unzip=False)

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    size = (180, 180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Define class names for binary classification
class_names = ['uninfected', 'parasitized']

# Main function to run the app
def main():
    # Download the model file
    download_model()
    
    # Load the model
    with st.spinner('Model is being loaded..'):
        model = load_model()

    # Streamlit UI
    st.write("""
             # Malaria Cell Classification
             """)

    # File uploader
    file = st.file_uploader("Please upload a cell image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        score = tf.nn.sigmoid(predictions[0])
        predicted_class = class_names[int(np.round(score))]
        st.write("Predicted Class: ", predicted_class)
        st.write("Confidence: ", np.max(score))

if __name__ == '__main__':
    main()
