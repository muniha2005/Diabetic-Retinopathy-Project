import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model (only once)
@st.cache_resource
def load_my_model():
    return load_model('Xception_model.h5')

model = load_my_model()

# Class labels
class_labels = ['mild', 'moderate', 'NO DR', 'Proliferate_DR', 'severe']

# --- Initialize session state for persistent storage ---
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'patients' not in st.session_state:
    st.session_state.patients = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''


# --- Signup Function ---
def signup():
    st.subheader("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Create Account"):
        if not username or not password or not confirm_password:
            st.error("Please fill out all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif username in st.session_state.users:
            st.error("Username already exists.")
        else:
            st.session_state.users[username] = password
            st.session_state.patients[username] = []
            st.success("Signup successful! Please login.")


# --- Login Function ---
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username not in st.session_state.users:
            st.error("User does not exist.")
        elif st.session_state.users[username] != password:
            st.error("Wrong password.")
        else:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")


# --- Upload Image and Predict ---
def upload_image():
    st.subheader("Upload Patient Image")
    patient_name = st.text_input("Patient Name")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and st.button("Predict"):
        if not patient_name:
            st.error("Please enter patient name.")
            return

        # Preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_labels[np.argmax(preds)]

        # Save patient info
        patient_info = {
            'name': patient_name,
            'image': uploaded_file.name,
            'prediction': predicted_class
        }

        st.session_state.patients[st.session_state.username].append(patient_info)

        # Display result
        st.image(img, caption="Uploaded Image", width=300)
        st.success(f"Prediction: **{predicted_class}**")


# --- View Patient History ---
def view_patients():
    st.subheader("Patient History")
    user_patients = st.session_state.patients.get(st.session_state.username, [])
    if not user_patients:
        st.info("No patient data found.")
    else:
        for patient in user_patients:
            st.markdown(f"**Name:** {patient['name']}")
            st.markdown(f"**Prediction:** {patient['prediction']}")
            st.markdown("---")


# --- Logout Function ---
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.success("You have been logged out.")


# --- Main UI ---
st.title("Diabetic Retinopathy Detector (Streamlit)")

# Set menu options based on login state
if st.session_state.logged_in:
    menu = ["Dashboard", "Upload", "Patients", "Logout"]
else:
    menu = ["Home", "Login", "Signup"]

choice = st.sidebar.selectbox("Menu", menu)

# Route logic
if choice == "Home":
    st.write("Welcome to the Diabetic Retinopathy Detector App!")

elif choice == "Signup":
    signup()

elif choice == "Login":
    login()

elif choice == "Dashboard":
    st.write(f"Welcome to your dashboard, **{st.session_state.username}**!")

elif choice == "Upload":
    upload_image()

elif choice == "Patients":
    view_patients()

elif choice == "Logout":
    logout()
