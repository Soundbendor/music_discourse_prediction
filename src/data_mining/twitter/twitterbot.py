import requests
import itertools
import tweepy
import os 

from time import sleep
from datetime import datetime
from dotenv import load_dotenv

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import Iterator

from src.data_mining.twitter.twitterbuilder import TwitterComment

class TwitterBot(CommentMiner):

    def __init__(self, f_key: str, search_depth: int = 10) -> None:
        self.api = self.auth_handler()
        self.twitter_epoch = datetime(2006, 3, 26)

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
        # 300 tweet/15 min limit per app. 1 app per academic license. 3 second delay.
        sleep(3)
        subs = self.api.search_all_tweets(query=self._build_query(song_name, artist_name),
                                          sort_order='relevancy',
                                          # TODO - determine optimial max results for retrieval
                                          max_results=10,
                                          start_time=self.twitter_epoch,
                                          expansions='referenced_tweets.id,author_id,in_reply_to_user_id,geo.place_id',
                                          tweet_fields='entities,geo,lang,public_metrics,conversation_id,created_at,\
                                              context_annotations,author_id,text',
                                          place_fields='country_code,name,geo,full_name,place_type',
                                          user_fields='username,location').data
        # TODO: Check Twitter status code here
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
            # TODO - lmao
            # TODO - when processing context enitity and annotation, handle type union elegantly 
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
        # TODO - change the logic here to maintain conversation structure by graph search.
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
        # TODO: Pull text language from twitter API, remove reference to l_detect
        return TwitterComment(
            id=str(tweet.id),
            body=tweet.text,
            lang=tweet.lang
            reply_count=tweet.public_metrics['reply_count'],
            score=tweet.public_metrics['like_count'],
            cashtags=[x['tag'] for x in tweet.entities.cashtags],
            hashtags=[x['tag'] for x in tweet.entities.hashtags],
            mentions=[x['username'] for x in tweet.entities.mentions],
            urls={k:v for k,v in tweet.entities.urls.items() if k in ['url', 'title', 'description', 'display_url']},
            context_annotations=itertools.chain([TwitterContextEntity(id=x.id, name=x.name, description=x.description) for k, v in tweet.context_annotations.items() if k == 'entitiy'][])
        )
