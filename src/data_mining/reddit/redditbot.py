import praw
import os
# dumb workaround for intellisense bug in vscode
from praw import models as praw_models
from dotenv import load_dotenv
from typing import Iterator, List
from data_mining.jsonbuilder import Submission, Comment
from data_mining.commentminer import CommentMiner


class RedditBot(CommentMiner):
    def __init__(self, f_key: str, search_depth: int = 10) -> None:
        load_dotenv()
        self.site_name = 'bot1'
        self.reddit = praw.Reddit(self.site_name,
                                  client_id=os.getenv('REDDIT_CLIENT_ID'),
                                  client_secret=os.getenv('REDDIT_CLIENT_SECRET'))
        self.search_depth = search_depth

    def process_submissions(self, submission: praw_models.Submission) -> Submission:
        s_lang = self.l_detect(f"{submission.title} {submission.selftext}")
        return Submission(
            title=submission.title,
            body=submission.selftext,
            lang=s_lang.lang,
            lang_p=s_lang.prob,
            url=submission.url,
            id=submission.id,
            score=submission.score,
            n_comments=submission.num_comments,
            subreddit=submission.subreddit.display_name,
            comments=list(map(self.process_comments,
                          self.get_comments(submission)))
        )

    def process_comments(self, comment: praw_models.Comment) -> Comment:
        c_lang = self.l_detect(comment.body)
        return Comment(
            id=comment.id,
            # Should this just be upvotes (t0 match youtube/twitter APIs)
            # put downvotes as a reddit data contract specific field?
            score=comment.score,
            body=comment.body,
            replies=len(comment.replies),
            lang=c_lang.lang,
            lang_p=c_lang.prob
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
        return list(post.comments)  # type: ignore
