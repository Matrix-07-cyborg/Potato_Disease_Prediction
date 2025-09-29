# 🍠 Potato Disease Prediction  

This project focuses on building a **machine learning model** to classify potato leaf diseases from images. Early detection of plant diseases is crucial in agriculture to minimize crop loss and improve yield.  

## 📌 Project Overview  
- Predicts whether a potato leaf is **healthy** or infected by diseases such as **Early Blight** or **Late Blight**.  
- Uses **image classification techniques** with deep learning.  
- Implemented in **Google Colab** for ease of training and experimentation.  

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries & Frameworks:**  
  - TensorFlow / Keras  
  - NumPy, Pandas  
  - Matplotlib, Seaborn (for visualization)  
- **Environment:** Google Colab  

## 📂 Dataset  
- The dataset consists of potato leaf images belonging to three categories:  
  - **Healthy**  
  - **Early Blight**  
  - **Late Blight**  
- Dataset is sourced from [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).  

## 🚀 Steps in the Project  
1. **Data Loading & Preprocessing**  
   - Image augmentation for better generalization  
   - Resizing and normalization of images  

2. **Model Building**  
   - Convolutional Neural Network (CNN) architecture  
   - Layers: Convolution, Pooling, Dropout, Fully Connected  

3. **Training & Evaluation**  
   - Trained with categorical cross-entropy loss  
   - Evaluated using **accuracy, confusion matrix, precision, recall**  

4. **Results**  
   - Achieved high accuracy in classifying potato leaf diseases  
   - Visualization of training vs validation performance  

## 📊 Model Performance  
- Training Accuracy: ~95–98%  
- Validation Accuracy: ~93–96%  
- Robust performance with minimal overfitting  

## 📸 Sample Predictions  
The model can successfully classify images into **Healthy**, **Early Blight**, or **Late Blight** categories.  

 
