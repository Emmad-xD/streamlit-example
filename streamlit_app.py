import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Define the labels
labels = ['samosa', 'hot_and_sour_soup']

# Load the model
learn = load_learner('model.pkl')

# Define the predict function
def predict(image):
    img = PILImage.create(image)
    pred, pred_idx, probs = learn.predict(img)
    return labels[pred_idx], probs[pred_idx]

# Create the Streamlit app
def app():
    st.title("Food Classifier")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Make a prediction if an image has been uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image.', use_column_width=True)
        label, prob = predict(image)
        st.write(f"Prediction: {label}; Probability: {prob:.4f}")

# Run the app
if __name__ == '__main__':
    app()
