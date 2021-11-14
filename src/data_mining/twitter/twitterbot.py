from dateutil.relativedelta import *
from time import sleep
import requests
import tweepy

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import Iterator,  List



class TwitterBot(CommentMiner):
    
    def __init__(self, f_key: str, search_depth: int = 10) -> None:
        self.api = self.auth_handler(f_key)


    def auth_handler(self, f_key: str) -> tweepy.Client:
        keys = self._process_api_key(f_key)
        return tweepy.Client(bearer_token = keys['CLIENT_INFO']['bearer_token'],
                        consumer_key = keys['CLIENT_INFO']['client_id'],
                        consumer_secret = keys['CLIENT_INFO']['client_secret'],
                        access_token = keys['ACCESS_TOKEN']['access_id'],
                        access_token_secret = keys['ACCESS_TOKEN']['access_secret'],
                        wait_on_rate_limit=True)


    def get_submissions(self, song_name: str, artist_name: str) -> Iterator[tweepy.Tweet]:
        # returns a list of top-level tweets mentioning the artist/track title
        # avoid the rate limit. 1 tweet/sec limit on historical queries. 
        sleep(15)
        return tweepy.Paginator(self.api.search_all_tweets,
                                self._build_query(song_name, artist_name),
                                max_results=500,
                                expansions='referenced_tweets.id,author_id,in_reply_to_user_id',
                                tweet_fields='entities,geo,lang,public_metrics,conversation_id,created_at',
                                user_fields='username').flatten()

    def _lookup_user(self, user_id) -> tweepy.Response:
        response = self.api.get_user(id = user_id)
        if isinstance(response, requests.models.Response):
            raise(Exception("Network Exception with Twitter API"))
        return response
        

    def process_submissions(self, p_tweet: tweepy.Tweet) -> Submission:
        s_lang = self.l_detect(p_tweet.text)
        user = self._lookup_user(p_tweet.author_id).data

        return Submission(
            title = "",
            body = p_tweet.text,
            lang = s_lang.lang,
            lang_p = s_lang.prob,
            url = f"https://twitter.com/{user.username}/status/{p_tweet.id}",
            id = str(p_tweet.id),
            score = p_tweet.public_metrics['like_count'],
            n_comments = p_tweet.public_metrics['reply_count'],
            subreddit = user.username,
            comments = list(map(self.process_comments, self.get_comments(p_tweet, user)))
        )


    def get_comments(self, p_tweet: tweepy.Tweet, user: tweepy.User) -> Iterator[tweepy.Tweet]:
        return tweepy.Paginator(self.api.search_all_tweets, f"conversation_id:{p_tweet.conversation_id} to:{user.username}",
                                    since_id = p_tweet.id, 
                                    max_results = 500,
                                    expansions = 'referenced_tweets.id,author_id,in_reply_to_user_id',
                                    tweet_fields='entities,geo,lang,public_metrics,conversation_id',
                                    user_fields='username').flatten()
            

    def process_comments(self, tweet: tweepy.Tweet) -> Comment:
        c_lang = self.l_detect(tweet.text)
        return Comment(
            id = str(tweet.id),
            score = tweet.public_metrics['like_count'],
            body = tweet.text,
            replies = tweet.public_metrics['reply_count'],
            lang = c_lang.lang,
            lang_p = c_lang.prob
        )

    
