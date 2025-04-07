import praw
import os
from dotenv import load_dotenv
import pandas as pd
import csv
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

load_dotenv()

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent="scrape"
)

# Define dataframe columns
columns = [
    "thread_id",
    "parent_post_title",
    "parent_post_body",
    "comment_id",
    "parent_comment_id",
    "comment_body",
    "comment_score",
    "timestamp",
    "author",
    "is_op",
    "summary"
]
df = pd.DataFrame(columns=columns)

# ----------------------
# Collect comments
# ----------------------
all_comments = []
comment_score_threshold = 25

for post in reddit.subreddit("MonsterHunter").top(limit=1):
    print(f"Title: {post.title}")
    print(f"Comments: {post.num_comments}\n")

    thread_id = post.id
    parent_post_title = post.title
    parent_post_body = post.selftext
    op_author = post.author.name if post.author else "[deleted]"

    post.comments.replace_more(limit=None)
    comments_list = post.comments.list()

    if not comments_list:
        print("No comments found for this post.")

    for comment in comments_list[:5]:  # Limit to 5 comments for testing
        comment_id = comment.id
        parent_comment_id = comment.parent_id.split("_")[-1]
        comment_body = comment.body.replace("\n", " ").strip()
        comment_score = comment.score
        timestamp = datetime.utcfromtimestamp(comment.created_utc).isoformat()
        author = comment.author.name if comment.author else "[deleted]"
        is_op = author == op_author

        all_comments.append({
            "thread_id": thread_id,
            "parent_post_title": parent_post_title,
            "parent_post_body": parent_post_body,
            "comment_id": comment_id,
            "parent_comment_id": parent_comment_id,
            "comment_body": comment_body,
            "comment_score": comment_score,
            "timestamp": timestamp,
            "author": author,
            "is_op": is_op,
            "summary": ""
        })

# ----------------------
# Topic Modeling for Summarization
# ----------------------
# Extract high-quality comments
documents = [c["comment_body"] for c in all_comments if c["comment_score"] > comment_score_threshold]
comment_scores = [c["comment_score"] for c in all_comments if c["comment_score"] > comment_score_threshold]

if documents:
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(2, 3),
        max_df=0.7,
        min_df=2,
        max_features=1000
    )
    X = vectorizer.fit_transform(documents)
    n_topics = min(5, len(documents))

    nmf = NMF(n_components=n_topics, random_state=42)
    X_reduced = nmf.fit_transform(X)
    terms = vectorizer.get_feature_names_out()

    # Get topic summaries
    topic_summaries = []
    for topic in nmf.components_:
        sorted_terms = sorted(zip(terms, topic), key=lambda x: x[1], reverse=True)[:10]
        topic_keywords = [term for term, _ in sorted_terms]
        topic_summaries.append(", ".join(topic_keywords))

    # Assign summary to high-quality comments
    for i, comment in enumerate(documents):
        topic_distribution = X_reduced[i]
        topic_idx = np.argmax(topic_distribution)
        summary = topic_summaries[topic_idx]

        # Find matching comment in all_comments and update
        for com in all_comments:
            if com["comment_body"] == comment and com["comment_score"] == comment_scores[i]:
                com["summary"] = summary
                break

# ----------------------
# Save to CSV
# ----------------------
df = pd.DataFrame(all_comments)

os.makedirs("Raw_Data", exist_ok=True)
df.to_csv("./Raw_Data/comments_detailed_with_topic_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

print("File saved with summarized comments.")
