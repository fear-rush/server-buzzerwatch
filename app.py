from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from sentiment import sentiment
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/twitter", methods=['POST'])
def result():
    data = request.json['hashtag']
    res = sentiment(data)
    responses = res.to_json(orient = "table")
    print(responses)
    return responses

if __name__ == '__main__':
    app.run(debug=True)