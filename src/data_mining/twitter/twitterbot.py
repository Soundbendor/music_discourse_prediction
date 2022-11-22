import tweepy
import os

from time import sleep
from datetime import datetime
from dotenv import load_dotenv
from pymongo.database import Database
from pymongo.results import InsertManyResult
from bson.objectid import ObjectId
from data_mining.commentminer import CommentMiner
from typing import List


class TwitterBot(CommentMiner):
    def __init__(self) -> None:
        self.client = self.auth_handler()
        self.twitter_epoch = datetime(2006, 3, 26)

    def auth_handler(self) -> tweepy.Client:
        load_dotenv()
        return tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_KEY_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            wait_on_rate_limit=True,
        )

    def process_submissions(self, db: Database, song: dict) -> List[ObjectId]:
        tl_tweets = self.persist(
            lambda: self._get_submissions(song["artist_name"], song["song_name"])
        )
        insert_response = self.make_transaction(
            db["posts"].insert_many, list(map(lambda x: x.data, tl_tweets))
        )
        if insert_response:
            tl_ids = insert_response.inserted_ids
            # accumulator list
            tweet_ids = tl_ids
            db["posts"].update_many(
                {"_id": {"$in": tl_ids}},
                {
                    "$set": {
                        "artist_name": song["artist_name"],
                        "song_name": song["song_name"],
                        "dataset": song["Dataset"],
                        "source": "Twitter",
                        "depth": 0,
                    },
                    "$rename": {
                        "text": "body",
                        "public_metrics.like_count": "score",
                        "public_metrics.reply_count": "n_replies",
                    },
                },
            )

            for i, tweet in enumerate(tl_tweets):
                replies = self.persist(lambda: self._get_comments(tweet))
                print(replies)
                if replies:
                    reply_insert_response = self.make_transaction(
                        db["posts"].insert_many, list(map(lambda x: x.data, replies))
                    )
                    #  reply_insert_response = db['posts'].insert_many(map(lambda x: x.data, replies))
                    if reply_insert_response:
                        reply_ids = reply_insert_response.inserted_ids

                        db["posts"].update_many(
                            {"_id": {"$in": reply_ids}},
                            {
                                "$set": {
                                    "artist_name": song["artist_name"],
                                    "song_name": song["song_name"],
                                    "dataset": song["Dataset"],
                                    "source": "Twitter",
                                    "depth": 1,
                                    "replies_to": tl_ids[i],
                                },
                                "$rename": {
                                    "text": "body",
                                    "public_metrics.like_count": "score",
                                    "public_metrics.reply_count": "n_replies",
                                },
                            },
                        )
                        # Update the top level post with a list of reply IDs.
                        db["posts"].update_one(
                            {"_id": tl_ids[i]}, {"$set": {"replies": reply_ids}}
                        )
                        tweet_ids += reply_ids
            return tweet_ids
        return []

    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        # returns a list of top-level tweets mentioning the artist/track title
        # 300 tweet/15 min limit per app. 1 app per academic license. 3 second delay.
        sleep(3)
        tweets = self.client.search_all_tweets(
            query=self._build_query(song_name, artist_name),
            sort_order="relevancy",
            # TODO - determine optimial max results for retrieval
            max_results=100,
            start_time=self.twitter_epoch,
            expansions="referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id",
            tweet_fields="entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text",
            place_fields="country_code,name,geo,full_name,place_type",
            user_fields="username,location",
        )
        # TODO: Check Twitter status code here
        # https://developer.twitter.com/en/support/twitter-api/error-troubleshooting
        print(tweets.errors)
        if tweets.errors:
            print(f"\n\nError: {tweets.errors}")
        if tweets.data:
            return tweets.data
        return []

    def _get_comments(self, p_tweet: tweepy.Tweet) -> List:
        # TODO - change the logic here to maintain conversation structure by graph search.
        sleep(3)
        tweets = self.client.search_all_tweets(
            query=f"conversation_id:{p_tweet.conversation_id}",
            start_time=self.twitter_epoch,
            sort_order="relevancy",
            max_results=100,
            expansions="referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id",
            tweet_fields="entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text",
            place_fields="country_code,name,geo,full_name,place_type",
            user_fields="username,location",
        )
        print(tweets.errors)
        if tweets.errors:
            print(f"\n\nError: {tweets.errors}")
        if tweets.data:
            return tweets.data
        return []
