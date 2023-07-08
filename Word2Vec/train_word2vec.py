import re
import os
import io
import gensim

from gensim.models import Word2Vec

def extract_text_from_directory(directory):
    texts = []
    i=0
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            i+=1
            text_path = os.path.join(directory, filename)
            with io.open(text_path, 'r', encoding="utf-8") as file:
                text= file.read()
            texts.append(text)
    return texts
	
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = text.split()
    return tokens

def preprocess_texts(texts):
    preprocessed_texts = []
    for text in texts:
        tokens = preprocess_text(text)
        preprocessed_texts.append(tokens)
    return preprocessed_texts

def train_language_model(sentences):
    model = Word2Vec(sentences, min_count=1)
    return model	

def save_language_model(model, model_path):
    model.save(model_path)
    
# dataset: https://github.com/Kmahmoudi/Tehran-Times-Dataset
text_directory = "path to data"

print("loading ...")

texts=extract_text_from_directory(text_directory)

print("processing ...")

preprocessed_texts = preprocess_texts(texts)

print("training model ...")
model = train_language_model(preprocessed_texts)

model_path = './model.wv'
save_language_model(model, model_path)

print("model saved successfully ! ")