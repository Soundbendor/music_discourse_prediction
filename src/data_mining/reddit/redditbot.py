from os import stat_result
import praw
import json
# dumb workaround for intellisense bug in vscode
from praw import models as praw_models
from configparser import ConfigParser
from typing import Iterator, List
from dataclasses import dataclass


class RedditBot():
    def __init__(self, keys: ConfigParser, search_depth: int = 10) -> None:
        self.site_name = 'bot1'
        self.reddit = praw.Reddit(self.site_name,
            client_id = keys['CLIENT_INFO']['client_id'],
            client_secret = keys['CLIENT_INFO']['client_secret'])
        self.search_depth = search_depth

    
    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        posts = []
        for post_index, submission in enumerate(self.get_submissions(song_name, artist_name)):
            post = Submission(
                index = post_index,
                title = submission.title,
                body = submission.selftext,
                url = submission.url,
                id = submission.id,
                score = submission.score,
                n_comments = submission.num_comments,
                subreddit = submission.subreddit.display_name
            )
            for comment_index, comment in enumerate(self.get_comments(submission)):
                post.comments.append(Comment(
                    index = comment_index,
                    id = comment.id,
                    score = comment.score,
                    body = comment.body,
                    replies = len(comment.replies)
                ))
            posts.append(post)
        return posts


    def get_submissions(self, song_name: str, artist_name: str) -> Iterator[praw_models.Submission]:
        subreddit = self.reddit.subreddit("all")
        
        return subreddit.search(self.build_query(song_name, artist_name),
            'top', 'lucene', "all", limit=self.search_depth)


    def build_query(self, song_name: str, artist_name: str) -> str:
        return f"title:\"{artist_name}\" \"{song_name} \""


    def get_comments(self, post: praw_models.Submission) -> List[praw_models.Comment]:
        post.comments.replace_more(limit=0)
        # WARN - type ignore because of apparant bug in praw stubs for pylance?
        return list(post.comments) #type: ignore



@dataclass
class Submission:
    index: int
    title: str
    body: str
    url: str
    id: str
    score: int
    n_comments: int
    subreddit: str
    comments: List[Comment] = []


@dataclass
class Comment:
    index: int
    id: str
    score: int
    body: str 
    replies: int
