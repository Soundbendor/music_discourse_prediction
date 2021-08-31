import praw
import configparser
# dumb workaround for intellisense bug in vscode
from praw import models as praw_models
from typing import Iterator, List
from data_mining.jsonbuilder import Submission, Comment
from data_mining.commentminer import CommentMiner


class RedditBot(CommentMiner):
    def __init__(self, f_key: str, search_depth: int = 10) -> None:
        keys = self._process_api_key(f_key)
        self.site_name = 'bot1'
        self.reddit = praw.Reddit(self.site_name,
            client_id = keys['CLIENT_INFO']['client_id'],
            client_secret = keys['CLIENT_INFO']['client_secret'])
        self.search_depth = search_depth

    
    def _process_api_key(self, f_key: str) -> configparser.ConfigParser:
        api_key = configparser.ConfigParser()
        api_key.read(f_key)
        return api_key


    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))


    def process_submissions(self, submission: praw_models.Submission) -> Submission:
        s_lang = self.l_detect(f"{submission.title} {submission.selftext}")
        return Submission(
                title = submission.title,
                body = submission.selftext,
                lang = s_lang.lang,
                lang_p = s_lang.prob,
                url = submission.url,
                id = submission.id,
                score = submission.score,
                n_comments = submission.num_comments,
                subreddit = submission.subreddit.display_name,
                comments = list(map(self.process_comments, self.get_comments(submission)))
            )


    def process_comments(self, comment: praw_models.Comment) -> Comment:
        c_lang = self.l_detect(comment.body)
        return Comment(
                id = comment.id,
                score = comment.score,
                body = comment.body,
                replies = len(comment.replies),
                lang = c_lang.lang,
                lang_p = c_lang.prob
            )


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

