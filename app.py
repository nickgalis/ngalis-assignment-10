import os
import pickle
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template, url_for
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import pandas as pd


app = Flask(__name__)


# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"
batch_size = 128
image_folder = 'coco_images_resized'


# Load the model and preprocess function
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()

tokenizer = get_tokenizer(model_name)


with open('../image_embeddings.pickle', 'rb') as f:
    embeddings_df = pd.read_pickle(f)

def search(query_embedding, k=5):
    scores = [
        (row['file_name'], F.cosine_similarity(query_embedding, torch.tensor(row['embedding']).to(device).unsqueeze(0)).item())
        for _, row in embeddings_df.iterrows()
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def get_text_embedding(text_query):
    text_tokens = tokenizer([text_query])
    return F.normalize(model.encode_text(text_tokens.to(device)))


def get_image_embedding(image_file):
    image = Image.open(image_file).convert('RGB')
    image_tensor = preprocess_val(image).unsqueeze(0).to(device)
    return F.normalize(model.encode_image(image_tensor))


def get_hybrid_embedding(text_query, image_file, weight):
    text_embedding = get_text_embedding(text_query)
    image_embedding = get_image_embedding(image_file)
    return F.normalize(weight * text_embedding + (1 - weight) * image_embedding)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        text_query = request.form.get('text_query', '').strip()
        image_file = request.files.get('image_query')
        
        weight = float(request.form.get('weight', 0.5))

        query_embedding = None

        if query_type == 'text' and text_query:
            query_embedding = get_text_embedding(text_query)
        elif query_type == 'image' and image_file:
            query_embedding = get_image_embedding(image_file)
        elif query_type == 'hybrid' and text_query and image_file:
            query_embedding = get_hybrid_embedding(text_query, image_file, weight)

        if query_embedding is not None:
            results = search(query_embedding)

    return render_template('index.html', results=results, image_folder=image_folder)

if __name__ == '__main__':
    app.run(port=5000, debug=True)