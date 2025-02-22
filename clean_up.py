import pandas as pd
# import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import csv

# nltk.download('punkt')
# nltk.download('wordnet')

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove non-alphabetic characters (keeps spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = text.strip()
    return text

def tokenize(text):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    return [token.lower() for token in tokens if token.isalpha()]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def process_text(text):
    tokens = tokenize(text)
    lemmatized_tokens = lemmatize(tokens)
    return lemmatized_tokens

def main():
    data_folder = 'Raw_Data'
    file_name = 'comments.csv'
    file_path = os.path.join(data_folder, file_name)
    
    processed_data_folder = 'Processed_Data'
    processed_file_path = os.path.join(processed_data_folder, 'processed_comments.csv')

    os.makedirs(processed_data_folder, exist_ok=True)

    if os.path.exists(processed_file_path):
        os.remove(processed_file_path)

    data = load_data(file_path)
    data['processed_text'] = data['comment_body'].apply(process_text)
    data[['comment_score', 'processed_text']].to_csv(processed_file_path, index=False, quoting=csv.QUOTE_MINIMAL)

if __name__ == "__main__":
    main()
