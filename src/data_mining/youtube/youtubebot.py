import pyyoutube

from .youtubeinterface import SubmissionInterface, YoutubeInterface
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
            title = str(sub_handler.snippet.title),
            body = str(sub_handler.snippet.description),
            lang = s_lang.lang,
            lang_p = s_lang.prob,
            url = sub_handler.get_url(),
            id = sub_handler.get_video_id(),
            score = sub_handler.get_video_score(),
            n_comments = int(sub_handler.stats.commentCount),
            subreddit = str(sub_handler.snippet.channelTitle),
            comments = list(map(self.process_comments, self.yt_client.get_comments(sub_handler.get_video_id())))
        )

    def process_comments(self, comment: pyyoutube.Comment) -> Comment:
        c_data = comment.snippet
        c_lang = self.l_detect(str(c_data.textOriginal))
        return Comment(
                id = str(c_data.parentId),
                score = int(c_data.likeCount),
                body = str(c_data.textOriginal),
                replies = 0,
                lang = c_lang.lang,
                lang_p = c_lang.prob
            )