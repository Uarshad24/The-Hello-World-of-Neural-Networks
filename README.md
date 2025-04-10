# The-Hello-World-of-Neural-Networks

Welcome to the "Hello World of Neural Networks" repository! This collection contains several beginner-friendly projects that demonstrate the basics of neural networks and deep learning.

### Projects
#### 1. House Pricing
This project focuses on predicting house prices based on various features, such as area, number of bedrooms, and other factors. It covers basic regression using neural networks.

Key Concepts:

  * Regression
  * Feature Scaling
  * Model Training and Evaluation

#### 2. Image Classification with TensorFlow (MNIST & Fashion MNIST)
This project explores image classification using two popular datasets: MNIST and Fashion MNIST. The MNIST dataset contains 60,000 28x28 grayscale images of handwritten digits (0-9) for training, with 10,000 images for testing. Fashion MNIST consists of 60,000 28x28 grayscale images of clothing items (e.g., T-shirts, pants, shoes) for training, and 10,000 test images. \
The objective is to build models to classify these images. For MNIST, a Dense Neural Network is used, while for Fashion MNIST, a Convolutional Neural Network (CNN) is applied to achieve better performance on image data. The project covers essential concepts like image preprocessing, model building, and training with TensorFlow. EarlyStopping is used to prevent overfitting during training.

Key Concepts

  * TensorFlow Built-in Datasets
  * Convolutional Neural Networks (CNNs)
  * Image Normalization
  * Early Stopping Callbacks
  * Model Training and Evaluation

#### 3. Image Classification using Real-World Images (CATS VS DOGS)
This project explores image classification using the Kaggle Cats vs Dogs dataset. Originally containing 25,000 images, the dataset has been filtered down to 2,002 images for this project. The goal is to build a model that can classify images as either containing a cat or a dog.

Key Concepts

  * Image Preprocessing with TensorFlow Data API
  * Data Augmentation
  * Transfer Learning (InceptionV3)
  * Model Training and Evaluation
  * Model Prediction

#### 4. Sentiment Analysis on IMDB Movie Reviews
This project focuses on classifying the sentiment of IMDB movie reviews as either positive or negative. It explores various deep learning architectures to handle natural language processing (NLP) tasks, including simple dense models, Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), and Convolutional Neural Networks (CNNs). The dataset consists of 50,000 movie reviews, with 25,000 for training and 25,000 for testing.

Key Concepts 

 * Text Preprocessing – Tokenization, padding, and word embeddings
 * Recurrent Neural Networks (RNNs) – LSTM and GRU for sequential data processing
 * Convolutional Neural Networks (CNNs) for NLP – Extracting local patterns in text
 * Embedding Layers – Representing words as dense vectors
 * Model Training and Evaluation

#### 5. Stock Price Forecasting with CNNs and LSTMs (Time Series Analysis)
This project focuses on forecasting Apple Inc. (AAPL) stock prices using historical data fetched from the Yahoo Finance API via the yfinance library. The model is trained on the Close price using a sliding window approach to capture time dependencies in the data. It combines Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory (LSTM) networks for sequence modeling.

The goal is to predict future stock prices based on past behavior. A windowed dataset is created, normalized using MinMax scaling, and passed to a hybrid CNN-LSTM model. The final model is trained and evaluated, and predictions are visualized to compare against real-world stock movements.

Key Concepts

 * Data extraction with yfinance – Retrieve historical AAPL stock price data directly from Yahoo Finance
 * Windowed Dataset – Transform historical data into sequences to feed into the model
 * Convolutional Neural Networks (CNNs) – Extract short-term trends and features from sequences
 * Recurrent Neural Networks (LSTM) – Learn temporal dependencies and long-term patterns
 * Model Training and Evaluation
 * Forecasting and Visualization – Generate future predictions and visualize actual vs predicted stock trends
