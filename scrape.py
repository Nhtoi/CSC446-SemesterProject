import praw
import os
from dotenv import load_dotenv
import requests

load_dotenv()  

reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_username = os.getenv("REDDIT_USERNAME")
reddit_password = os.getenv("REDDIT_PASSWORD")


auth = requests.auth.HTTPBasicAuth(reddit_client_id, reddit_client_secret)


data = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    username=reddit_username,
    password=reddit_password,
    user_agent="scrape"
)


headers = {
    "User-Agent": "MyAPI/0.1 by Nhtoi",
}

res = requests.post(
    "https://www.reddit.com/api/v1/access_token",
    auth=auth,
    data={
        "grant_type": "password",
        "username": reddit_username,
        "password": reddit_password
    },
    headers=headers
)


TOKEN = res.json().get("access_token")
if not TOKEN:
    raise ValueError("Failed to get access token from Reddit API")

headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

# print(headers)


