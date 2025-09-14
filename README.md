🍔 Fast Food Image Classification using CNN

📌 Project Description:

This project focuses on classifying fast food images using Convolutional Neural Networks (CNNs) and transfer learning with pretrained models. The models used include MobileNetV2, ResNet50, DenseNet121, InceptionV3, and a fine-tuned MobileNet. The system is trained on a labeled fast food dataset and evaluated using accuracy, precision, recall, and F1-score. The best-performing model is deployed using Streamlit, allowing real-time classification of uploaded food images.

🚀 Features:

Classification of multiple fast food categories (e.g., burger, pizza, fries, etc.)

Implementation of five pretrained CNN architectures

Data preprocessing and augmentation for robust training

Performance evaluation using standard ML metrics

Web app deployment with Streamlit for real-time predictions

🛠️ Technologies Used:

Python

TensorFlow / Keras

Streamlit

NumPy, Pandas

Matplotlib, Seaborn

PIL / OpenCV

Scikit-learn

📂 Project Structure:

data/ → Dataset (organized into train/validation/test)

models/ → Saved trained models

notebooks/ → Training and experiment notebooks

app.py → Streamlit app for deployment

ann.py → Training script for CNN models

requirements.txt → Dependencies list

README.md → Project documentation

📊 Results:

The project compared five pretrained models. DenseNet121 achieved the best overall performance, while MobileNetV2 provided the best trade-off between accuracy and speed, making it suitable for mobile deployment.

🌐 Deployment:

The best model is deployed using Streamlit. Users can upload an image, and the app predicts the fast food category in real-time. The project can also be extended for mobile and cloud deployment.

📖 Future Work:

Expand dataset with more diverse food categories

Optimize models using quantization and pruning

Implement ensemble approaches for higher accuracy

Add calorie estimation and nutrition analysis

✨ Acknowledgements:

Pretrained models from TensorFlow/Keras Applications

Dataset sourced from Kaggle Fast Food Image Dataset

Research inspiration from existing CNN-based food classification studies
