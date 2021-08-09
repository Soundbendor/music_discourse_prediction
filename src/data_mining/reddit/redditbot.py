import praw
# dumb workaround for intellisense bug in vscode
from praw import models as praw_models

class RedditBot():
    def __init__(self) -> None:
        self.site_name = 'bot1'
        self.reddit = praw.Reddit(self.site_name)


    def query(self, song_name: str, artist_name: str) -> None:
        pass


    def get_submissions(self, query: str, search_depth: int) -> list:
        subreddit = self.reddit.subreddit("all")
        return subreddit.search(query, 'top', 'lucene', "all", limit=search_depth)


    def get_comments(self, post: praw_models.Submission):
        post.comments.replace_more(limit=0)
        return list(post.comments)