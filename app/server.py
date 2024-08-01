from flask import Flask, request, jsonify
from predict import predict_sentiment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    prediction = predict_sentiment(text)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return jsonify({'sentiment': sentiment, 'score': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
