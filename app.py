import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model from pickle file
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Image preprocessing function
def preprocess_image(img):
    img = image.load_img(img, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale the image (same as in ImageDataGenerator during training)
    img_array = img_array / 255.0
    return np.vstack([img_array])


# Add background image via CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://w.wallhaven.cc/full/ex/wallhaven-ex8l8k.png');  /* Set your background image */
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            
        }
    </style>
""", unsafe_allow_html=True)


# Frontend for Streamlit
st.title("Sentiment Detection from Images")
st.write("Upload an image and the model will predict whether you seem happy or sad!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    # Display progress bar
    progress = st.progress(0)

    # Preprocess the image
    img = uploaded_file
    image_data = preprocess_image(img)

    # Predict sentiment
    with st.spinner('Predicting...'):
        progress.progress(50)  # Halfway progress
        prediction = model.predict(image_data)
        progress.progress(100)  # Complete progress

    # Display results with appropriate label based on prediction
    if prediction[0] < 0.5:
        st.success("😊 You seem happy!")
    else:
        st.warning("😢 You seem sad!")
    
    # Optionally, you can display the prediction value for debugging purposes
    #st.write(f"Prediction value: {prediction[0][0]:.4f}")
    st.write("The model predicts if the emotion in the image is more aligned with 'happy' or 'sad'.")

# Sidebar for extra features
#st.sidebar.header("About the Model")
#st.sidebar.write(
    """
    This sentiment detection model is based on a Convolutional Neural Network (CNN). 
    The model has been trained to classify images as either showing a 'happy' or 'sad' sentiment. 
    It uses basic CNN layers to extract features from the image and make predictions.
    """
#

# Option to see the model structure
#if st.sidebar.button("View Model Summary"):
    #st.sidebar.text(model.summary())
