import json

from itertools import chain
from typing import Dict, List, Tuple, Union

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from dataclasses import dataclass, field


scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
api_service_name = 'youtube'
api_version = 'v3'


@dataclass
class YoutubeSearchResult:
    snippet: dict
    video: dict

class YoutubeInterface:

    def __init__(self, key: str):
        self.api = self._process_api_key(key)


    def _process_api_key(self, f_key: str) -> googleapiclient.discovery.Resource:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(f_key, scopes)
        creds = flow.run_console()
        return googleapiclient.discovery.build(api_service_name, api_version, credentials=creds)

    # Returns instances of Search resources
    # https://developers.google.com/youtube/v3/docs/search/list
    def search_by_keywords(self, query: str, limit: int) -> List[YoutubeSearchResult]:
        return list(filter(None, map(self.get_videos, self._search_keyword(query, limit))))

    def get_videos(self, s_result: dict) -> Union[YoutubeSearchResult, None]:
        try:
            v_id = s_result['id']['videoId']
        except KeyError:
            return None
        v_resource = self.get_video_by_id(v_id)
        return YoutubeSearchResult(snippet=s_result, video=v_resource)

    def _search_keyword(self, query: str, limit: int) -> List[Dict]:
        return self.api.search().list(
            part='snippet',
            maxResults=limit,
            q=query
        ).execute()['items']


    # Returns list of CommentThread resources
    # https://developers.google.com/youtube/v3/docs/commentThreads#resource 
    def _get_comment_threads(self, video_id) -> List[Dict]:
        try:
            return self.api.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            ).execute()['items']
        except googleapiclient.errors.HttpError:
            return []
    
    # Returns list of Comment resources
    # https://developers.google.com/youtube/v3/docs/comments#resource
    def get_comments(self, video_id: str) -> List[Dict]:
        return list(chain.from_iterable(map(self._flatten_threads, self._get_comment_threads(video_id))))

    def _flatten_threads(self, thread: dict) -> List[Dict]:
        if int(thread['snippet']['totalReplyCount']) < 1 or not thread['replies']:
            return [thread['snippet']['topLevelComment']]
        thread['replies']['comments'].insert(0, thread['snippet']['topLevelComment'])
        return thread['replies']['comments']
            

    # returns list of Video resources
    # https://developers.google.com/youtube/v3/docs/videos/list#resource
    def get_video_by_id(self, video_id) -> List[Dict]:
        return self.api.videos().list(
            part='snippet,contentDetails,statistics',
            id=video_id
        ).execute()['items'][0]
