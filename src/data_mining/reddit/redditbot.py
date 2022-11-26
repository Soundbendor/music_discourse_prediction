import praw
import os

from dotenv import load_dotenv
from itertools import chain
from praw.models.reddit.comment import CommentForest
from praw.reddit import Comment, Submission
from database.driver import Driver
from bson.objectid import ObjectId
from typing import List
from data_mining.commentminer import CommentMiner

SITE_NAME = "bot1"


class RedditBot(CommentMiner):
    def __init__(self, _: str) -> None:
        load_dotenv()
        self.reddit = self._authenticate()

    def _authenticate(self) -> praw.Reddit:
        return praw.Reddit(
            SITE_NAME,
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="Music Discourse Scraper - Oregon State University EECS",
        )

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        submissions = self._get_submissions(song["song_name"], song["artist_name"])
        # WARN - We are ignoring information about the Submission itself (including submission text)
        # to preserve schema structure
        comments = list(chain.from_iterable(map(self._get_comments, submissions)))
        return db.insert_posts(
            comments,
            {
                "artist_name": song["artist_name"],
                "song_name": song["song_name"],
                "dataset": song["Dataset"],
                "source": "Reddit",
            },
            # mongodb gets fussy when we try to pass an empty rename body
            {"subreddit_name_prefixed": "subreddit_name"},
            {k: f"$$this.{k}" for k, v in comments[0].items()},
        )

    def _get_submissions(self, song_name: str, artist_name: str) -> List[Submission]:
        return self.reddit.subreddit("all").search(
            self.build_query(song_name, artist_name), "top", "lucene", "all", limit=100
        )

    def build_query(self, song_name: str, artist_name: str) -> str:
        return f'title:"{artist_name}" "{song_name} "'

    def _replace_comments(self, c: CommentForest) -> List[dict]:
        _ = c.replace_more(0)
        return list(map(self._get_attributes, list(c)))

    def _get_attributes(self, c: Comment) -> dict:
        return {
            k: v
            for k, v in dict(vars(c)).items()
            if not callable(v)
            and not k.startswith("_")
            and k != "author"
            and k != "subreddit"
        }

    def _parse_comment(self, comment: Comment) -> dict:
        c = self._get_attributes(comment)
        c["replies"] = self._replace_comments(comment.replies)
        c["submission"] = comment._submission.id
        c["subreddit"] = comment.subreddit.name
        try:
            c["author"] = comment.author.name
        except AttributeError:
            c["author"] = ""
        return c

    # WARN - need to wrap in a persist call, as replace_more requires multiple api requests
    def _get_comments(self, post: Submission) -> List[dict]:
        return list(map(self._parse_comment, post.comments))  # type: ignore
