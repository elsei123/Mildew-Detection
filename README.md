# [Cherry Leaf Mildew Detector](https://midetector-86d4ac524814.herokuapp.com/)

The [**Cherry Leaf Mildew Detector**](https://midetector-86d4ac524814.herokuapp.com/) is an AI-based solution designed for sustainable agriculture. The project aims to automatically and accurately identify whether a cherry leaf is healthy or infected with powdery mildew, enabling quick and effective decision-making to prevent crop losses.

---

## Table of Contents

- [Project Requirements](#project-requirements)
- [Features](#features)
- [Data and Model](#data-and-model)
- [Dashboard Details](#dashboard-details)
- [Ethical Considerations and NDA](#ethical-considerations-and-nda)
- [Future Improvements](#future-improvements)

---

## Project Requirements

- **Visual Analysis:**  
  Perform visual studies to distinguish healthy leaves from those affected by powdery mildew.

- **Automated Prediction:**  
  Develop a machine learning system using a Convolutional Neural Network (CNN) that predicts if a leaf is healthy or infected, with a minimum target accuracy of 97%.

- **Interactive Dashboard:**  
  Provide a dashboard that offers:
  - A project summary detailing the dataset and client requirements.
  - Visual analyses (mean images, variability, and image montages) to highlight differences between classes.
  - A prediction interface that allows image uploads, displays prediction results, and offers CSV download options.
  - Technical details of the model and its performance metrics.
  - Information on ethical considerations and data confidentiality (NDA).

- **Integration with External Data:**  
  Include a link for downloading the [dataset](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) to facilitate access to the images.

---

## Features

- **Interactive Dashboard:**  
  Built with Streamlit, the dashboard enables you to:
  - Understand the project's context and the benefits of early mildew detection.
  - Upload images for prediction, view results with graphs and tables, and download prediction data.
  - Analyze data distribution and monitor the evolution of the model's accuracy.
  - Access detailed visual analysis including:
    - **Mean Image:** Represents the average visual characteristics of each class.
    - **Variability:** Highlights internal variation among images.
    - **Montage:** Displays a collage of sample images from each class.
  - Review technical details of the model and its training metrics.
  - Explore the Ethics & NDA section, reinforcing the confidentiality of the data.

- **Machine Learning Model:**  
  A CNN built with TensorFlow/Keras for binary classification (healthy vs. powdery mildew), enhanced by data augmentation techniques.

- **Result Download:**  
  Users can download prediction results in CSV format for further analysis.

---

## Data and Model

### Dataset

The dataset consists of cherry leaf images organized into two categories:
- **healthy:** Images of healthy cherry leaves.
- **powdery_mildew:** Images of leaves infected with powdery mildew.

### Model

- **Architecture:**  
  The model employs a Convolutional Neural Network (CNN) featuring:
  - **Convolution and Pooling Layers:** For feature extraction and dimensionality reduction.
  - **Flatten Layer:** To convert activation maps into a vector.
  - **Dense Layers with Dropout:** To prevent overfitting.
  - **Output Layer:** A sigmoid-activated layer for binary classification.

- **Training:**  
  The model is trained with:
  - **Data Augmentation:** Using `ImageDataGenerator` to expand the dataset.
  - **Data Split:** 80% for training and 20% for validation.
  - **Configuration:** 10 epochs, the Adam optimizer, and Binary Crossentropy loss.
  - **Target:** Achieve at least 97% validation accuracy.

- **Model Saving and Loading:**  
  After training, the model is saved to `src/model.h5` and loaded with caching to ensure efficient use in the dashboard.

---

## Dashboard Details

### Home
- **Project Overview:**  
  Introduces the project, outlining the importance of early mildew detection and its benefits for sustainable agriculture.

### Prediction
- **Image Upload & Analysis:**  
  Users can upload cherry leaf images. The system resizes and displays the images, performs predictions, and shows probability graphs for each class.
- **Result Table:**  
  Displays image names and prediction results, with an option to download the table as a CSV file.

### Analysis
- **Graphical Visualizations:**  
  Provides charts that illustrate:
  - The distribution of predictions (bar chart).
  - The proportion of healthy versus infected leaves (pie chart).
  - The evolution of model accuracy over time (line chart).

### Findings
- **Detailed Visual Study:**  
  Presents in-depth visual analyses:
  - **Mean Image:** Shows the average appearance of each class.
  - **Variability:** Highlights internal variations (standard deviation) within each class.
  - **Montage:** Displays a collage of sample images to represent each class.

### Hypothesis
- **Approach & Validation:**  
  Outlines the project hypothesis, explaining how combining visual analysis with CNNs can effectively differentiate the classes, and details the validation strategy to meet the performance target.

### Technical
- **Model Details:**  
  Provides comprehensive information about the model architecture, training procedures, and performance metrics, including a summary of the model.

### Ethics & NDA
- **Data Confidentiality:**  
  Explains that the dataset is provided under a Non-Disclosure Agreement (NDA), with strict measures in place to ensure data confidentiality and restrict access to authorized personnel only.

### Download Link
- **Dataset Access:**  
  A sidebar link directs users to download the cherry leaf images from an external source, [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

---

## Ethical Considerations and NDA

The data used in this project is provided under a Non-Disclosure Agreement (NDA) and is strictly confidential. Access is restricted to authorized personnel only, and sharing images or results with unauthorized third parties is prohibited. Robust security measures are in place to ensure compliance with the NDA terms.

---

## Future Improvements

- **Real Metric Integration:**  
  Update graphs and performance comparisons with actual training results.
  
- **Quantitative Dataset Analysis:**  
  Include detailed statistics, such as the total number of images, class distribution, and variability analysis.
  
- **Model Optimization:**  
  Explore alternative architectures and techniques to further improve accuracy and reduce the model size for easier deployment.
  
- **Dashboard Enhancements:**  
  Improve interactivity and usability by providing more detailed user feedback.
  
- **Expanded Documentation:**  
  Offer more comprehensive documentation of the data pipeline, training procedures, and ethical practices.
