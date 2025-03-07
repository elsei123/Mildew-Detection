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
