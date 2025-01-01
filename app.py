import flask
import pandas as pd
import tensorflow as tf
import torch
import transformers
import sklearn
import nltk
import PIL

from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import os
import re
import random
import math
import io
import base64

app = Flask(__name__)
nltk.download('stopwords')
nltk.download('wordnet')

data_path = os.path.join(os.path.dirname(__file__), 'training.1600000.processed.noemoticon.csv')
data = pd.read_csv(data_path, encoding="latin1", header=None)
data.columns = ["sentiment", "id", "date", "query", "user", "text"]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

data['clean_text'] = data['text'].apply(clean_text)
data['sentiment'] = data['sentiment'].replace({0: "negative", 4: "positive"})

X = data['clean_text']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    return f1_score(y_test, y_pred, pos_label='positive', zero_division=0)

lr_f1 = evaluate_model(lr_model, X_test_vec, y_test)
nb_f1 = evaluate_model(nb_model, X_test_vec, y_test)

if lr_f1 > nb_f1:
    best_model = lr_model
else:
    best_model = nb_model

device_index = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device_index
)

def detailed_sentiment_analysis(tweet):
    result = sentiment_pipeline(tweet)[0]
    sentiment_label = result['label'].lower()
    score = result['score']
    if sentiment_label == "positive":
        if score > 0.9:
            return "Happiness"
        elif 0.7 < score <= 0.9:
            return "Excitement"
        else:
            return "Hope"
    elif sentiment_label == "negative":
        if score > 0.9:
            return "Anger"
        elif 0.7 < score <= 0.9:
            return "Fear"
        else:
            return "Sadness"
    else:
        if score > 0.9:
            return "Informative"
        elif 0.7 < score <= 0.9:
            return "Explanatory"
        else:
            return "Neutral"

def create_abstract_art(tweet, sentiment):
    width, height = 600, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    color_palettes = {
        "Happiness": [(255, 223, 186), (255, 181, 71), (255, 140, 0)],
        "Excitement": [(252, 92, 101), (253, 150, 68), (254, 211, 48)],
        "Hope": [(174, 214, 241), (133, 193, 233), (93, 173, 226)],
        "Sadness": [(64, 64, 122), (102, 102, 153), (153, 153, 204)],
        "Anger": [(179, 57, 57), (204, 87, 87), (230, 115, 115)],
        "Fear": [(43, 62, 80), (67, 97, 128), (99, 140, 186)],
        "Informative": [(178, 235, 242), (128, 222, 234), (77, 182, 172)],
        "Explanatory": [(255, 236, 179), (255, 213, 79), (255, 193, 7)],
        "Neutral": [(224, 224, 224), (189, 189, 189), (158, 158, 158)]
    }
    palette = color_palettes.get(sentiment, [(200, 200, 200)])
    start_color = random.choice(palette)
    end_color = random.choice(palette)
    for x in range(width):
        ratio = x / width
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        draw.line([(x, 0), (x, height)], fill=(r, g, b))
    shape_count = random.randint(10, 20)
    for _ in range(shape_count):
        shape_type = random.choice(["circle", "polygon", "arc", "spiral", "wave", "lines"])
        color = random.choice(palette)
        if shape_type == "circle":
            x0, y0 = random.randint(0, width), random.randint(0, height)
            radius = random.randint(30, 80)
            x1, y1 = x0 + radius, y0 + radius
            draw.ellipse([x0, y0, x1, y1], fill=color)
        elif shape_type == "polygon":
            points = []
            for _ in range(random.randint(4, 6)):
                px = random.randint(0, width)
                py = random.randint(0, height)
                points.append((px, py))
            draw.polygon(points, fill=color)
        elif shape_type == "arc":
            x0, y0 = random.randint(0, width), random.randint(0, height)
            x1, y1 = x0 + random.randint(50, 150), y0 + random.randint(50, 150)
            start_angle = random.randint(0, 360)
            end_angle = start_angle + random.randint(30, 270)
            draw.arc([x0, y0, x1, y1], start_angle, end_angle, fill=color, width=3)
        elif shape_type == "spiral":
            center_x, center_y = width // 2, height // 2
            spiral_points = random.randint(50, 150)
            spiral_radius = random.randint(20, 50)
            rotation_factor = random.choice([1, -1])
            for i in range(spiral_points):
                angle = 0.1 * i * rotation_factor
                current_radius = spiral_radius + i * 0.6
                sx = int(center_x + current_radius * math.cos(angle))
                sy = int(center_y + current_radius * math.sin(angle))
                draw.ellipse([sx, sy, sx+3, sy+3], fill=color)
        elif shape_type == "wave":
            wave_height = random.randint(20, 60)
            step = random.randint(5, 15)
            offset = random.randint(0, 360)
            for x_coord in range(0, width, step):
                y_coord = int(height / 2 + wave_height * math.sin(2 * math.pi * (x_coord + offset) / 100))
                draw.ellipse([x_coord, y_coord, x_coord+6, y_coord+6], fill=color)
        elif shape_type == "lines":
            x_start = random.randint(0, width)
            y_start = random.randint(0, height)
            line_length = random.randint(50, 150)
            for i in range(5):
                draw.line(
                    [(x_start, y_start + i*5), (x_start + line_length, y_start + i*5)],
                    fill=color, width=2
                )
    font = ImageFont.load_default()
    text_x, text_y = 10, height - 30
    draw.text((text_x, text_y), f"Tweet: {tweet[:50]}...", fill="black", font=font)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/random_tweet')
def random_tweet():
    random_row = data.sample(n=1).iloc[0]
    tweet = random_row['text']
    sentiment = detailed_sentiment_analysis(tweet)
    img = create_abstract_art(tweet, sentiment)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return render_template('index.html', tweet_text=tweet, sentiment=sentiment, image_data=img_base64)

@app.route('/manual_tweet', methods=['POST'])
def manual_tweet():
    user_tweet = request.form.get('user_tweet', '')
    sentiment = detailed_sentiment_analysis(user_tweet)
    img = create_abstract_art(user_tweet, sentiment)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return render_template('index.html', tweet_text=user_tweet, sentiment=sentiment, image_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
