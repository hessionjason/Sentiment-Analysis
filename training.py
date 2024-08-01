import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data
sentences = [
    'I love this product',
    'This is the worst thing I ever bought',
    'I am so happy with this',
    'I hate this item',
]

labels = [1, 0, 1, 0]

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='post')

# Model creation
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(padded_sequences, np.array(labels), epochs=10)

# Save the model
model.save("sentiment_model.h5")#h5 file is a file that stores large information . used to save modals, efficient storage
