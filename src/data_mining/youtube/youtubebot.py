import pyyoutube

from .youtubeinterface import YoutubeInterface
from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment

from typing import List





scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


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
        response_data = YoutubeInterface.get_submission_snippet(submission) 
        video = self.yt_client.get_video_by_id(video_id=submission.id.videoId)
        s_lang = self.l_detect(f"{response_data.title} {response_data.description}")
        return Submission(
            title = str(response_data.title),
            body = str(response_data.description),
            lang = s_lang.lang,
            lang_p = s_lang.prob,
            url = submission.url,
            id = submission.id.videoId,
            score = (video.statistics.likeCount - video.statistics.dislikeCount),
            n_comments = video.statistics.commentCount,
            subreddit = response_data.channelTitle,
            #comments = list(map(self.process_comments, self.get_comments(submission)))
        )