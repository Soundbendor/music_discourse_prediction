import requests
import itertools
import tweepy
import os 
import pymongo
import bson

from time import sleep
from datetime import datetime
from dotenv import load_dotenv

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import Iterator, List

from .twitterbuilder import TwitterComment

class TwitterBot(CommentMiner):

    def __init__(self) -> None:
        self.client = self.auth_handler()
        self.twitter_epoch = datetime(2006, 3, 26)

    def auth_handler(self) -> tweepy.Client:
        load_dotenv()
        return tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                             consumer_key=os.getenv('TWITTER_API_KEY'),
                             consumer_secret=os.getenv('TWITTER_API_KEY_SECRET'),
                             access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
                             access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
                             wait_on_rate_limit=True)

    # Requirements
    # (yt, reddit) Pull list of top level submissions
    # Get all top level comments
    # Graph search all replies
    # Use depth key to track conversation structure
    # Enforce fixed fields across sources
    # insert all of these fields into their respective collections (Submissions, Comments)
    # Add type information (source(twitter, etc.), date retrieved/updated) as field
    # Append submission, comment, reply object IDs to return list
    # All references will be stored in Submissions in the Songs collection

    # Expects that source documeng had "artist_name" and "song_name" fields.
    def process_submissions(self, db: pymongo.database.Database, song: dict) -> List[bson.objectid.ObjectId]:
        tl_tweets = self.get_submissions(song['artist_name'], song['song_name'])
        tweets = [self.get_comments(tweet) for tweet in tl_tweets]
        insert_response = db['posts'].insert_many(map(lambda x: x.data, tl_tweets))
        tl_ids = insert_response.inserted_ids
        db['posts'].update_many({'_id': {'$in': tl_ids}},
                                {'$set': {
                                    'artist_name': song['artist_name'],
                                    'song_name': song['song_name'],
                                    'dataset': song['Dataset'], },
                                 '$rename': {
                                    'text': 'body',
                                    'public_metrics.like_count': 'score',
                                    'public_metrics.reply_count': 'n_replies'
                                     }
                                 })
        


    def get_submissions(self, song_name: str, artist_name: str) -> List:
        # returns a list of top-level tweets mentioning the artist/track title
        # 300 tweet/15 min limit per app. 1 app per academic license. 3 second delay.
        sleep(3)
        tweets = self.client.search_all_tweets(query=self._build_query(song_name, artist_name),
                                          sort_order='relevancy',
                                          # TODO - determine optimial max results for retrieval
                                          max_results=10,
                                          start_time=self.twitter_epoch,
                                          expansions='referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id',
                                          tweet_fields='entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text',
                                          place_fields='country_code,name,geo,full_name,place_type',
                                          user_fields='username,location')
        # TODO: Check Twitter status code here
        # https://developer.twitter.com/en/support/twitter-api/error-troubleshooting
        print(tweets.errors)
        if tweets.errors['response_code'] != 200:
            print(f"\n\nError: {tweets.errors}")
        # Concat includes and tweet data to preserve username and referenced tweet IDs
        tweets = list(map(lambda x: dict(x.data, **x.includes), tweets))
        if tweets:
            return tweets
        return []


    def get_comments(self, p_tweet: tweepy.Tweet) -> List:
        sleep(3)
        # TODO - change the logic here to maintain conversation structure by graph search.
        comments = self.client.search_all_tweets(query=f"conversation_id:{p_tweet.conversation_id}",
                                              since_id=self.twitter_epoch,
                                              max_results=100,
                                              expansions='referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id',
                                              tweet_fields='entities,geo,lang,public_metrics,conversation_id,created_at,context_annotations,author_id,text',
                                              place_fields='country_code,name,geo,full_name,place_type',
                                              user_fields='username,location')
        if comments:
            return comments
        return []
