import pandas as pd
import numpy as np

def get_sample_data():
    """
    Returns a DataFrame with sample data for visualizations.
    """
    data = pd.DataFrame({
        'Month': range(1, 13),
        'Value': np.random.randint(50, 150, 12)
    })
    data = data.set_index('Month')
    return data
