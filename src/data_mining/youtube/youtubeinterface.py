
import pyyoutube
import json

from itertools import chain
from typing import List, Type

from pyyoutube.api import Api
from pyyoutube.models.search_result import SearchResultId, SearchResultSnippet

# a wrapper for a wrapper for a wrapper of the youtube API. 
# fml. - aidan
baseURL = 'https://youtu.be/zV9T42c1yWk'

scopes = ["https://www.googleapis.com/auth/youtube",
          "https://www.googleapis.com/auth/youtube.channel-memberships.creator",
          "https://www.googleapis.com/auth/youtube.force-ssl",
          "https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtubepartner",
          "https://www.googleapis.com/auth/youtubepartner-channel-audit"]


class YoutubeInterface:

    def __init__(self, key: str):
        self.api = self._process_api_key(key)


    def _process_api_key(self, f_key: str) -> pyyoutube.Api:
        with open(f_key, 'r') as config_raw:
            cfg_json = json.load(config_raw)
            api = pyyoutube.Api(
                    client_id=cfg_json['installed']['client_id'],
                    client_secret=cfg_json['installed']['client_secret']
                )
            print(f"Click the following link to log in.\n{api.get_authorization_url(scope=scopes)}")
            response = input("Enter the authorization code:")
            api.generate_access_token(authorization_response=response)
            return api

    def search_by_keywords(self, query: str, search_type: list, count: int, limit: int) -> List[pyyoutube.SearchResult]:
        return self.api.search_by_keywords(q=query, search_type=search_type, count=count, limit=limit).items


    def _get_video_responses_by_id(self, video_id) -> List[pyyoutube.Video]:
        return self.api.get_video_by_id(video_id=video_id).items

    def _get_comment_threads(self, video_id) -> List[pyyoutube.CommentThread]:
        return self.api.get_comment_threads(video_id=video_id, count=None).items
    
    def get_comments(self, video_id: str) -> List[pyyoutube.Comment]:
        return list(chain.from_iterable(map(self._flatten_comments, self._get_comment_threads(video_id))))


    def _flatten_comments(self, thread: pyyoutube.CommentThread) -> List[pyyoutube.Comment]:
        snippet = thread.snippet
        replies = thread.replies
        top_comment = snippet.topLevelComment
        replies.insert(0,top_comment)
        return replies
            

    def get_video_by_id(self, video_id) -> pyyoutube.Video:
        videos = self._get_video_responses_by_id(video_id)
        if len(videos) > 0:
            return videos[0]
        raise(APIResponseError("No videos found for the given ID."))


class SubmissionInterface:

    def __init__(self, submission: pyyoutube.SearchResult, api: YoutubeInterface):
        self.submission = submission
        self.snippet = submission.snippet
        self.video = api.get_video_by_id(self.get_video_id())
        self.stats = self._get_video_statistics()


    def get_video_id(self) -> str:
        v_id: SearchResultId = self.submission.id
        return v_id.videoId


    def _get_video_statistics(self) -> pyyoutube.VideoStatistics:
        return self.video.statistics


    def get_video_score(self) -> int:
        return int(self.stats.likeCount) - int(self.stats.dislikeCount)


    def get_url(self) -> str:
        return baseURL + self.get_video_id()


class APIResponseError(Exception):
    pass