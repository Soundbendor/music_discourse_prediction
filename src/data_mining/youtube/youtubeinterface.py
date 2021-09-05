
import pyyoutube
import json

from typing import List, Type

from pyyoutube.api import Api
from pyyoutube.models.search_result import SearchResultId, SearchResultSnippet

# a wrapper for a wrapper for a wrapper of the youtube API. 
# fml. - aidan
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
            print(f"Click the following link to log in.\n{api.get_authorization_url()}")
            response = input("Enter the authorization code:")
            api.generate_access_token(authorization_response=response)
            return api

    def search_by_keywords(self, query: str, search_type: list, count: int, limit: int) -> List[pyyoutube.SearchResult]:
        return self.validate_return(
            self.api.search_by_keywords(q=query, search_type=search_type, count=count, limit=limit).items, 
            List[pyyoutube.SearchResult])

    def get_video_by_id(self, video_id) -> List[pyyoutube.Video]:
        return self.validate_return(
            self.api.get_video_by_id(video_id=video_id).items,
            List[pyyoutube.Video]
        )
            
    def validate_return(self, item, d_type: Type):
        if type(item) == d_type:
            return d_type(item)
        raise(APIResponseError)

    @staticmethod
    def get_submission_snippet(submission) -> SearchResultSnippet:
        if type(submission) == pyyoutube.SearchResultSnippet:
            return submission
        raise(APIResponseError)

    @staticmethod
    def get_video_id(submission) -> str:
        if submission.id == None:
            raise APIResponseError
        else:
            v_id: SearchResultId = SearchResultId(submission)
            return str(v_id.videoId)

class APIResponseError(Exception):
    pass