import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load and filter comments
documents = []
with open('./Processed_Data/processed_comments.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        upvotes = int(row[0])
        if upvotes > 25:
            clean_text = " ".join(eval(row[1])).replace("'", "")  # Convert list string to normal text
            documents.append(clean_text)

if documents:
    # Vectorize with optimized settings
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(2,3), 
        max_df=0.7, 
        min_df=2, 
        max_features=1000
    )
    X = vectorizer.fit_transform(documents)

    # Apply NMF for better topic separation
    n_topics = min(5, len(documents))
    nmf = NMF(n_components=n_topics, random_state=42)
    X_reduced = nmf.fit_transform(X)

    # Extract key phrases
    terms = vectorizer.get_feature_names_out()

    for i, topic in enumerate(nmf.components_):
        sorted_terms = sorted(zip(terms, topic), key=lambda x: x[1], reverse=True)[:10]
        topic_keywords = [term for term, _ in sorted_terms]
        
        print(f"\nConcept {i}:")
        print("This topic is about:", ", ".join(topic_keywords))
else:
    print("No comments met the upvote threshold.")
