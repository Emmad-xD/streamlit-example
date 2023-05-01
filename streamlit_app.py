import streamlit as st
from fastai.vision.all import *

def GetLabel(fileName):
  return fileName.split('-')[0]

learn = load_learner('path/to/model.pkl')

def app():
  st.set_page_config(page_title="Food Classifier", page_icon=":pizza:")

  st.title("Food Classifier")
  st.write("Upload an image of either samosa or nachos to get a prediction")

  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    label, _, probs = learn.predict(image)
    st.write(f"This is a {label}.")
    st.write(f"{labelA} {probs[1].item():.6f}")
    st.write(f"{labelB} {probs[0].item():.6f}")

if __name__ == "__main__":
  app()
