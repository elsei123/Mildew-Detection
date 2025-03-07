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

    