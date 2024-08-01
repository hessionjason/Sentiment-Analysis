import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
model = tf.keras.models.load_model('sentiment_model.h5')

# Tokenizer settings (must match training settings)
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')
    prediction = model.predict(padded_sequences)
    return prediction[0][0]

# Example usage
if __name__ == "__main__":
    text = "I am so happy"
    print(predict_sentiment(text))
