import pyyoutube
import json
from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import Dict, Iterator, List

from pyyoutube.models.search_result import SearchResultSnippet
from pyyoutube.models.video import VideoListResponse




scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


class YoutubeBot(CommentMiner):


    def __init__(self, key: str) -> None:
        self.yt_client = self._process_api_key(key)

    def _process_api_key(self, f_key: str) -> pyyoutube.Api:
        with open(f_key, 'r') as config_raw:
            cfg_json = json.load(config_raw)
            api = pyyoutube.Api(
                    client_id=cfg_json['installed']['client_id'],
                    client_secret=cfg_json['installed']['client_secret']
                )
            print(f"Click the following link to log in.\n{api.get_authorization_url()}")
            response = input("Enter the authorization code:")
            api.generate_access_token(authorization_response=response)
            return api

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))


    def get_submissions(self, song_name: str, artist_name: str) -> List[pyyoutube.SearchResult]:
        # Pylance bug - Because of weak union typing in pyyoutube library, typehint does not work here
        # search_by_keywords() returns a SearchListResponse object with a parameter `items` of type List[SearchResult] 
        return self.yt_client.search_by_keywords(q=self._build_query(song_name, artist_name),
            search_type=['video'], count=self.search_depth, limit=self.search_depth).items #type: ignore


    def _build_query(self, song_name: str, artist_name: str) -> str:
        return f"\"{artist_name}\" \"{song_name}\""

    def process_submissions(self, submission: pyyoutube.SearchResult) -> Submission:
        response_data: SearchResultSnippet = submission.snippet #type: ignore
        video: VideoListResponse = self.yt_client.get_video_by_id(submission.id.videoId)
        s_lang = self.l_detect(f"{response_data.title} {response_data.description}")
        return Submission(
            title = response_data.title,
            body = response_data.description,
            lang = s_lang.lang,
            lang_p = s_lang.prob,
            url = submission.url,
            id = submission.id.videoId,
            score = (video.statistics.likeCount - video.statistics.dislikeCount),
            n_comments = video.statistics.commentCount,
            subreddit = response_data.channelTitle,
            #comments = list(map(self.process_comments, self.get_comments(submission)))
        )