import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_sample_data
from src.model import load_trained_model
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_trained_model():
    """
    Loads and returns the trained model from the absolute path using caching to avoid reloading each time.
    """
    model_path = os.path.abspath("src/model.h5")
    if not os.path.exists(model_path):
        st.error(f"Error: Model not found at {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def predict_image(model, image):
    """
    Receives the model and a preprocessed image, and performs the prediction.
    """
    image = image.reshape((1, *image.shape))
    prediction = model.predict(image)
    return prediction

def get_sample_data():
    """
    Returns a DataFrame with sample data for visualizations.
    """
    data = pd.DataFrame({
        'Month': range(1, 13),
        'Value': np.random.randint(50, 150, 12)
    })
    return data.set_index('Month')



# Configure Streamlit page layout
st.set_page_config(page_title="Mildew Detection Dashboard", layout="wide")

# CSS style
st.markdown("""
<style>
    /* Default style for light mode */
    .sub-title { font-size: 26px; color: #2c3e50; margin-bottom: 20px; text-align: center; }
    .section-title { font-size: 22px; color: #34495e; margin-top: 20px; }
    .content { font-size: 18px; color: #2c3e50; line-height: 1.5; }
    .legend { font-size: 16px; color: #7f8c8d; text-align: center; }
    
    /* Styles for dark mode */
    @media (prefers-color-scheme: dark) {
      .sub-title { color: #f0f0f0 !important; }
      .section-title { color: #e0e0e0 !important; }
      .content { color: #f0f0f0 !important; }
      .legend { color: #c0c0c0 !important; }
    }
</style>
""", unsafe_allow_html=True)

# Set the page title
st.title("🍒 Cherry Leaf Mildew Detector 🍃")

# Sidebar menu for navigation
menu = st.sidebar.radio(
    "📌 Navigation", ["🏠 Home", "📸 Prediction", "📊 Analysis"])

# Home Page
if menu == "🏠 Home":
    st.markdown("<h2 class='sub-title'>🌿 Artificial Intelligence for Sustainable Farming</h2>",
                unsafe_allow_html=True)
    st.markdown("""
        <p class='content'>
            The <b>Cherry Leaf Mildew Detector</b> uses <b>artificial intelligence</b> to detect <span class='highlight'>powdery mildew</span> at an early stage, 
            a fungal disease that can compromise the entire cherry harvest. With our technology, farmers can take quick action, 
            preventing losses and optimizing crop yield.
        </p>
        <h2 class='section-title'>🦠 What is Powdery Mildew?</h2>
        <p class='content'>
            <b>Powdery mildew</b> is a disease caused by the fungus <i>Podosphaera clandestina</i>, which spreads rapidly in humid environments.
            It forms a <span class='highlight'>white powdery layer</span> on the leaves, affecting plant growth and drastically reducing production.
        </p>
        <h2 class='section-title'>🧐 How to Identify the Symptoms?</h2>
        <p class='content'>
            ✅ Small <b>white spots</b> start appearing on younger leaves.<br>
            ✅ Leaves may <b>deform and curl</b> as the infection progresses.<br>
            ✅ In advanced stages, the fungus spreads to both sides of the leaf.<br>
            🌡 The disease thrives in <b>humid climates and excessive irrigation</b>.
        </p>
        <h2 class='section-title'>🔬 How Does Artificial Intelligence Work?</h2>
        <p class='content'>
            Our AI analyzes leaf images and determines with high accuracy whether the tree is <span class='highlight'>healthy or infected</span>.
            This enables efficient monitoring and more effective preventive actions.
        </p>
        <h2 class='section-title'>💼 Benefits for Farmers</h2>
        <p class='content'>
            ✅ <b>Automated Monitoring</b>: Reduces time spent on manual inspections.<br>
            ✅ <b>Smart Use of Fungicides</b>: Application only when necessary, reducing costs.<br>
            ✅ <b>Higher Profitability</b>: Effective protection against harvest losses.
        </p>
    """, unsafe_allow_html=True)

    # Prediction Page
elif menu == "📸 Prediction":
    st.header("📌 Make a Prediction")
    st.markdown("""
    <p class='content'>
        In this section, you can upload images of cherry leaves for the artificial intelligence model to analyze
        and determine whether the leaf is <b>healthy</b> or <b>infected with powdery mildew</b>. 
        Early detection can help in disease prevention and control, improving agricultural production quality.
    </p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("🖼️ Select images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model = load_trained_model()
        threshold = 0.5
        results = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_resized = image.resize((256, 256))
            st.image(image_resized, caption="Uploaded Image (256x256 px)",use_column_width=False)

            img_array = np.array(image_resized)
            if img_array.shape[-1] == 4:
                img_array = img_array[:, :, :3]
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            label = "Healthy 🍃" if prediction < threshold else "Powdery Mildew ⚠️"
            confidence = float(prediction)
            results.append({"Image": uploaded_file.name, "Class": label, "Confidence": confidence})

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(["Healthy", "Powdery Mildew"], [1 - confidence, confidence], color=["green", "red"])
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_title("Leaf Classification", fontsize=14)
            ax.tick_params(axis='both', labelsize=10)
            st.pyplot(fig)
            
            if label == "Healthy 🍃":
                st.write("<p style='color: green; text-align: center; font-size: 20px;'>No anomalies detected in the leaf.</p>", unsafe_allow_html=True)
            else:
                st.write("<p style='color: red; text-align: center; font-size: 20px;'>Powdery mildew detected on the leaf.</p>", unsafe_allow_html=True)
                
        st.write("🔍 **Prediction Results**")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

# Analysis Page
elif menu == "📊 Analysis":
    st.header("📈 Model Analysis")
    
    labels = ["Healthy", "Powdery Mildew"]
    values = [1200, 900]
    
    st.subheader("📊 Bar Chart - Prediction Distribution")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(labels, values, color=["green", "red"])
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.set_title("Prediction Distribution", fontsize=12)
    ax.tick_params(axis='both', labelsize=8)
    st.pyplot(fig)
    st.caption("Distribution of predictions between healthy and infected leaves")
    
    st.subheader("🟠 Pie Chart - Classification Proportion")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=["green", "red"])
    ax.set_title("Proportion of Healthy vs. Powdery Mildew", fontsize=12)
    st.pyplot(fig)
    st.caption("Proportion of healthy vs. infected leaves")

    st.subheader("📉 Line Chart - Accuracy History")
    history_file = "../jupyter_notebooks/history.npy"
    if os.path.exists(history_file):
        history_data = np.load(history_file, allow_pickle=True).item()
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(history_data["accuracy"], label="Training", linestyle='-', marker='o')
        ax.plot(history_data["val_accuracy"], label="Validation", linestyle='-', marker='s')
        ax.set_title("Accuracy Evolution", fontsize=12)
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        st.caption("Model accuracy history during training")
    else:
        dummy_history = {"accuracy": [0.7, 0.8, 0.85, 0.9], "val_accuracy": [0.65, 0.75, 0.8, 0.85]}
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(dummy_history["accuracy"], label="Training", linestyle='-', marker='o')
        ax.plot(dummy_history["val_accuracy"], label="Validation", linestyle='-', marker='s')
        ax.set_title("Accuracy Evolution", fontsize=12)
        ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        st.caption("Model accuracy history during training (sample data)")

