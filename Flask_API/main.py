import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle
import os

app = Flask(__name__)
CORS(app)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def preprocess_comment(comment):
    try:
        if not isinstance(comment, str):
            comment = str(comment) if pd.notna(comment) else ""
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = " ".join([words for words in comment.split() if words not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in Preprocessing comment : {e}")
        return comment


def get_model_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    candidates = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(repo_root, filename)
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(repo_root, filename)


def load_model(model_path, vectorizer_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    with open(vectorizer_path, 'rb') as file:
        loaded_vectorizer = pickle.load(file)
    return loaded_model, loaded_vectorizer


model, vectorizer = None, None
try:
    model, vectorizer = load_model(
        get_model_path('lgbm_model.pkl'),
        get_model_path('tfidf_vectorizer.pkl')
    )
except Exception as e:
    print(f"Notice: Model or vectorizer not loaded at startup: {e}")


def ensure_model_loaded():
    global model, vectorizer
    if model is None or vectorizer is None:
        model, vectorizer = load_model(
            get_model_path('lgbm_model.pkl'),
            get_model_path('tfidf_vectorizer.pkl')
        )


@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json or {}
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        ensure_model_loaded()
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


@app.route("/predict_with_timestamps", methods=['POST'])
def predict_with_timestamps():
    data = request.json or {}
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400
    try:
        ensure_model_loaded()
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        predictions = model.predict(dense_comments).tolist()
        predictions = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


@app.route("/generate_chart", methods=['POST'])
def generate_chart():
    try:
        data = request.get_json() or {}
        sentiment_counts = data.get("sentiment_counts") or data.get("Sentiment_counts")

        if not sentiment_counts:
            return jsonify({"error": "No Sentiment counts provided"}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={"color": 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        plt.close()
        app.logger.error(f"Error in /generate_chart: {str(e)}")
        return jsonify({"error": f"Chart generation failed : {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json() or {}
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        if WordCloud is None:
            return jsonify({"error": "WordCloud is not installed in this environment"}), 501

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = " ".join(preprocessed_comments)
        if not text.strip():
            text = "No words available"

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud : {e}")
        return jsonify({"error": f"Word cloud generation failed : {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json() or {}
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        df.set_index('timestamp', inplace=True)

        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: "Positive"}

        try:
            monthly_counts = df.resample("ME")['sentiment'].value_counts().unstack(fill_value=0)
        except ValueError:
            monthly_counts = df.resample("M")['sentiment'].value_counts().unstack(fill_value=0)

        monthly_totals = monthly_counts.sum(axis=1)

        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        monthly_percentages = monthly_percentages.fillna(0)

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {
            -1: "red",
            0: "gray",
            1: "green"
        }
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )
        plt.title("Monthly sentiment percentage over time")
        plt.xlabel('Month')
        plt.ylabel("Percentage of comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        plt.close()
        app.logger.error(f"Error in /generate_trend_graph : {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    