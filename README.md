# Sentiment-Analysis
Real-Time Sentiment Analysis for Customer Feedback Using Neural Networks and Streamlit App
# Approach
I have scrapped the data needed for sentiment analysis from IMDB webpage using selenium.
Pre processed the data and added the classes in reference with the ratings they provided.
To handle the imbalance, I used data augmentation - synonym augmentation and adding minority class to maatch the majority class
Tokenized and padded the text for building the bidirectional LSTM model.
Created a streamlit app to connect the model build and predict the user reviews.
# Tools and Technologies:
Programming Language : Python
Frameworks and Libraries: TensorFlow/Keras or PyTorch (for neural network development).
Hugging Face Transformers I tried both BERT and Distil BERT.
Streamlit (for building the web application).
Deployment Platforms: AWS.
Open-source platform: Streamlit Community Cloud.
Datasets: Scraped IMDB webpage data using selenium
