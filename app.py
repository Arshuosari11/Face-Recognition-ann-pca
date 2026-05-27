import streamlit as st
from main import recognize_face

st.title("AI Face Recognition System")

uploaded_file = st.file_uploader("Upload a Face Image" , type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    with open("temp.jpg", "wb")as f:
        f.write(uploaded_file.getbuffer())
    st.image("temp.jpg", caption="Uploaded Image")
    result = recognize_face("temp.jpg")
    st.success(f"Predicted Person: {result}")
        