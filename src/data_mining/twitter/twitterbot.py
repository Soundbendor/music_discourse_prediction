import requests
import tweepy
import os 

from dateutil.relativedelta import *
from time import sleep
from dotenv import load_dotenv

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import Iterator


class TwitterBot(CommentMiner):

    def __init__(self, f_key: str, search_depth: int = 10) -> None:
        self.api = self.auth_handler()

    def auth_handler(self) -> tweepy.Client:
        load_dotenv()
        return tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
                             consumer_key=os.getenv('TWITTER_CLIENT_ID'),
                             consumer_secret=os.getenv('TWITTER_CLIENT_SECRET'),
                             access_token=os.getenv('TWITTER_ACCESS_ID'),
                             access_token_secret=os.getenv('TWITTER_ACCESS_SECRET'),
                             wait_on_rate_limit=True)

    def get_submissions(self, song_name: str, artist_name: str) -> Iterator[tweepy.Tweet]:
        # returns a list of top-level tweets mentioning the artist/track title
        # avoid the rate limit. 1 tweet/sec limit on historical queries.
        sleep(3)
        subs = self.api.search_all_tweets(query=self._build_query(song_name, artist_name),
                                          max_results=10,
                                          expansions='referenced_tweets.id,author_id,in_reply_to_user_id',
                                          tweet_fields='entities,geo,lang,public_metrics,conversation_id,created_at',
                                          user_fields='username').data
        if subs:
            return subs
        return []

    def _lookup_user(self, user_id) -> tweepy.Response:
        response = self.api.get_user(id=user_id)
        if isinstance(response, requests.models.Response):
            raise(Exception("Network Exception with Twitter API"))
        return response

    def process_submissions(self, p_tweet: tweepy.Tweet) -> Submission:
        s_lang = self.l_detect(p_tweet.text)
        user = self._lookup_user(p_tweet.author_id).data

        return Submission(
            title="",
            body=p_tweet.text,
            lang=s_lang.lang,
            lang_p=s_lang.prob,
            url=f"https://twitter.com/{user.username}/status/{p_tweet.id}",
            id=str(p_tweet.id),
            score=p_tweet.public_metrics['like_count'],
            n_comments=p_tweet.public_metrics['reply_count'],
            subreddit=user.username,
            comments=list(map(self.process_comments,
                          self.get_comments(p_tweet, user)))
        )

    def get_comments(self, p_tweet: tweepy.Tweet, user: tweepy.User) -> Iterator[tweepy.Tweet]:
        sleep(3)
        comments = self.api.search_all_tweets(query=f"conversation_id:{p_tweet.conversation_id} to:{user.username}",
                                              since_id=p_tweet.id,
                                              max_results=100,
                                              expansions='author_id,in_reply_to_user_id',
                                              tweet_fields='entities,geo,lang,public_metrics,conversation_id',
                                              user_fields='username').data
        if comments:
            return comments
        return []

    def process_comments(self, tweet: tweepy.Tweet) -> Comment:
        c_lang = self.l_detect(tweet.text)
        return Comment(
            id=str(tweet.id),
            score=tweet.public_metrics['like_count'],
            body=tweet.text,
            replies=tweet.public_metrics['reply_count'],
            lang=c_lang.lang,
            lang_p=c_lang.prob
        )
