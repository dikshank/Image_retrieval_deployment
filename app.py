import os
from flask import Flask, render_template, request
import numpy as np
import io
from PIL import Image
import image_encoder
import cv2
import text_encoder
import tensorflow as tf
from scipy.sparse import load_npz
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer

# import nltk
# import multiprocessing as mp
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import spacy
import re
# from nltk.stem import WordNetLemmatizer
# import nltk
from transformers import TFBertModel, BertTokenizer

app = Flask(__name__)

# Function to render the basic HTML template
FEATURE_1_ENABLED = os.environ.get("FEATURE_1_ENABLED", "false").lower() == "true"
@app.route('/')
def home():
    title = "Feature 1" if FEATURE_1_ENABLED else "Base Version"
<<<<<<< HEAD
    print("FEATURE_1_ENABLED:", FEATURE_1_ENABLED)
=======
>>>>>>> 5e3efbc90aefdc675cad00d380c21513a0d38261
    return render_template('index.html', title=title or "Default Title")
# "C:\Users\Lenovo\Desktop\IMAGE RETRIEVAL DEPLOYMENT\image_encoder_weights.h5"
# Function to handle the form submission and process the text and images
# with open("tokenizer.pkl", 'rb') as f:
#     tokenizer = pickle.load(f)

img_encoder_weights_path =  "image_encoder_weights.h5"
image_encoder_model = image_encoder.create_image_encoder()
image_encoder_model.load_weights(img_encoder_weights_path)
print(image_encoder_model.summary())

# # load the sparse matrix from file
# sparse_embedding_matrix = load_npz('embedding_matrix.npz')
# # convert the sparse matrix back to a numpy array
# embedding_matrix = sparse_embedding_matrix.toarray()

text_encoder_weights_path = "text_encoder_distiltbert_weights.h5"
# text_encoder_weights = np.load('text_encoder_weightsfloat16.npz')
text_encoder_model = text_encoder.create_text_encoder()
# text_encoder_model.set_weights( [w.astype(np.float32) for w in text_encoder_weights.values()])
text_encoder_model.load_weights(text_encoder_weights_path)
# text_encoder_model.set_weights([text_encoder_weights[f'weight_{i}'] for i in range(len(text_encoder_weights))])
# print(text_encoder_model.summary())

img_norm_mean = np.array([0.485,0.456,0.456])
img_norm_std = np.array([0.229,0.224,0.225])
bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
bert_layer = TFBertModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Load the English language model in spaCy
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop]
    tokens = " ".join(tokens)
    return [tokens]

def image_to_array(image_array,img_norm_mean=img_norm_mean,img_norm_std=img_norm_std):
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (100, 100))
    image = image.astype(float) / 255.0
    image = (image - img_norm_mean)/img_norm_std
    image = np.array(image)
    return image

@app.route('/process_form', methods=['POST'])
def process_form():
    text = request.form['text']
    image_files = request.files.getlist('image')
    images = []
    # image_embeddings = []
    for image_file in image_files:
        img_bytes = io.BytesIO(image_file.read())
        image_array = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        image = image_to_array(image_array=image_array)
        images.append(image)
        # img = Image.open(img_bytes)
        # img_array = np.array(img)
        # img_tensor = tf.convert_to_tensor(img_array)
    print(np.array(images).shape)
    image_embeddings = image_encoder_model(np.array(images))
    
    tokens = preprocess_text(text)
    tokens = bert_tokenizer(tokens)['input_ids']
    print(tokens)
    padded_tokens = pad_sequences(tokens, maxlen=30)
    print(padded_tokens)
    text_embedding = bert_layer(padded_tokens)[0]
    print(f'text_embedding shape {text_embedding.shape}')
    text_embedding = text_encoder_model(text_embedding)
    # text_embedding = text_embedding[:, :32]
    # image_embeddings.append(img_embedding)
    # text_embedding = text_encoder.encode_text(text)
    # Process the text and images here
    # ...
    # emb_shape = np.array(text_embedding).shape
    similarity = cosine_similarity(image_embeddings, text_embedding)

    # return render_template('result.html', text=text_embedding, image_embeddings=image_embeddings)
    # return render_template('result.html', embedding_shape = text_embedding, similarity = similarity) 
    return render_template('result.html', text_embedding = text_embedding, image_embeddings = image_embeddings, similarity= similarity)
# Function to display the result page with the processed data
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
