import tweepy
import requests
import os

from time import sleep
from datetime import datetime
from dotenv import load_dotenv
from bson.objectid import ObjectId
from data_mining.commentminer import CommentMiner
from database.driver import Driver
from typing import List, Callable


class TwitterBot(CommentMiner):
    def __init__(self, _: str) -> None:
        self.client = self._authenticate()
        self.twitter_epoch = datetime(2006, 3, 26)

    def authenticate(self) -> tweepy.Client:
        load_dotenv()
        return tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_KEY_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            wait_on_rate_limit=True,
        )

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        tweets = self._persist(
            lambda: self._get_submissions(song["artist_name"], song["song_name"])
        )

        return db.insert_posts(
            tweets,
            {
                "artist_name": song["artist_name"],
                "song_name": song["song_name"],
                "dataset": song["Dataset"],
                "source": "Twitter",
            },
            {
                "text": "body",
                "public_metrics.like_count": "score",
                "public_metrics.reply_count": "n_replies",
            },
        )

    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        # 300 tweet/15 min limit per app. 1 app per academic license. 3 second delay.
        sleep(3)
        tweets: tweepy.Response = self.client.search_all_tweets(
            query=self._build_query(song_name, artist_name),
            sort_order="relevancy",
            max_results=100,
            start_time=self.twitter_epoch,
            expansions="referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id",
            tweet_fields="entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text",
            place_fields="country_code,name,geo,full_name,place_type",
            user_fields="username,location",
        )  # type: ignore

        # https://developer.twitter.com/en/support/twitter-api/error-troubleshooting
        if tweets.errors:
            print(f"\n\nError: {tweets.errors}")
        if tweets.data:
            return list(map(self._insert_replies, tweets.data))
        return []

    def _insert_replies(self, tweet: tweepy.Tweet) -> tweepy.Tweet:
        tweet.data["replies"] = self._get_comments(tweet)
        return tweet.data

    def _get_comments(self, p_tweet: tweepy.Tweet) -> List:
        sleep(3)
        tweets: tweepy.Response = self.client.search_all_tweets(
            query=f"conversation_id:{p_tweet.conversation_id}",
            start_time=self.twitter_epoch,
            sort_order="relevancy",
            max_results=100,
            expansions="referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id",
            tweet_fields="entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text",
            place_fields="country_code,name,geo,full_name,place_type",
            user_fields="username,location",
        )  # type: ignore

        if tweets.errors:
            print(f"\n\nError: {tweets.errors}")
        if tweets.data:
            return list(map(lambda x: x.data, tweets))
        return []

    def _persist(self, func: Callable[[], List], retries: int = 3):
        for _ in range(0, retries):
            try:
                return func()
            except requests.exceptions.ConnectionError:
                print("Twitter: Connection Error! Reconnecting in 5...")
                sleep(5)
        raise RuntimeError("Exceeded maximum retries")
