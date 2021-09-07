import pyyoutube

from .youtubeinterface import SubmissionInterface, YoutubeInterface
from .youtubeinterface import lookup
from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment

from typing import List

class YoutubeBot(CommentMiner):


    def __init__(self, key: str, search_depth: int = 10) -> None:
        self.yt_client = YoutubeInterface(key)
        self.search_depth = search_depth

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))


    def get_submissions(self, song_name: str, artist_name: str) -> List[pyyoutube.SearchResult]:
        return self.yt_client.search_by_keywords(query=self._build_query(song_name, artist_name),
            search_type=['video'], count=self.search_depth, limit=self.search_depth)


    def _build_query(self, song_name: str, artist_name: str) -> str:
        return f"\"{artist_name}\" \"{song_name}\""


    def process_submissions(self, submission: pyyoutube.SearchResult) -> Submission:
        sub_handler = SubmissionInterface(submission, self.yt_client)
        s_lang = self.l_detect(f"{sub_handler.snippet.title} {sub_handler.snippet.description}")
        return Submission(
            title = lookup(sub_handler.snippet.title, str),
            body = lookup(sub_handler.snippet.description, str),
            lang = s_lang.lang,
            lang_p = s_lang.prob,
            url = sub_handler.get_url(),
            id = sub_handler.get_video_id(),
            score = sub_handler.get_video_score(),
            n_comments = lookup(sub_handler.stats.commentCount, int),
            subreddit = lookup(sub_handler.snippet.channelTitle, str),
            comments = list(map(self.process_comments, self.yt_client.get_comments(sub_handler.get_video_id())))
        )

    def process_comments(self, comment: pyyoutube.Comment) -> Comment:
        c_data = lookup(comment.snippet, pyyoutube.CommentSnippet)
        c_lang = self.l_detect(lookup(c_data.textOriginal, str))
        return Comment(
                id = lookup(c_data.parentId, str),
                score = lookup(c_data.likeCount, int),
                body = lookup(c_data.textOriginal, str),
                replies = 0,
                lang = c_lang.lang,
                lang_p = c_lang.prob
            )