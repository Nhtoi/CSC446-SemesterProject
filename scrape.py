import praw
import os
from dotenv import load_dotenv
import pandas as pd
import csv  # Import csv module for quoting options

load_dotenv()
df = pd.DataFrame(columns=["comment_body", "comment_score"])

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent="scrape"
)

for post in reddit.subreddit("MonsterHunter").top(limit=1):
    print(f"Title: {post.title}")
    print(f"Comments: {post.num_comments}\n")

    post.comments.replace_more(limit=None)  
    comments_list = post.comments.list()  
    if not comments_list:
        print("No comments found for this post.")

    for comment in comments_list[:5]:  # Limit to 5 comments for testing
        df = df._append({
            "comment_body": comment.body.replace("\n", " "),  # Remove newlines
            "comment_score": comment.score,
        }, ignore_index=True)

# Save to CSV while ensuring text remains on one line
df.to_csv("./Raw_Data/comments.csv", index=False, quoting=csv.QUOTE_MINIMAL)  # Use minimal quoting to keep formatting clean
