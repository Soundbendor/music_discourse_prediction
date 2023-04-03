import time
import os
from google.oauth2.credentials import Credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from googleapiclient.errors import HttpError
from itertools import chain
from bson.objectid import ObjectId
from googleapiclient.discovery import Resource
from data_mining.commentminer import CommentMiner
from typing import Callable, Dict, List
from database.driver import Driver

BASE_URL = "http://youtu.be/"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"


class YoutubeBot(CommentMiner):
    def __init__(self, key: str) -> None:
        self.yt_client = self._authenticate(key)

    def _authenticate(self, f_key: str) -> Resource:
        if os.path.exists("yt_token.json"):
            creds = Credentials.from_authorized_user_file("yt_token.json", SCOPES)
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(f_key, SCOPES)
            creds = flow.run_local_server(port=0)
            with open("yt_token.json", "w") as token:
                token.write(creds.to_json())
        return googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, credentials=creds)

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        videos = self._get_submissions(song["song_name"], song["artist_name"])
        comments = list(chain.from_iterable(map(self._get_comments, videos)))
        return db.insert_posts(
            comments,
            {
                "artist_name": song["artist_name"],
                "song_name": song["song_name"],
                "dataset": song["Dataset"],
                "source": "Youtube",
            },
            {
                "snippet.textOriginal": "body",
                "snippet.likeCount": "score",
            },
            {
                "body": "$$this.snippet.textOriginal",
                "id": "$$this.id",
                "snippet": "$$this.snippet",
                "score": "$$this.snippet.likeCount",
            },
        )

    def _build_query(self, song_name: str, artist_name: str) -> str:
        return '"{}" "{}"'.format(
            artist_name.replace('"', "").replace("_", ""), song_name.replace('"', "").replace("_", "")
        )

    # Returns a list of Video objects
    def _get_submissions(self, song_name: str, artist_name: str) -> List[dict]:
        return self._persist(
            lambda: self._get_videos_by_id(
                list(
                    map(
                        lambda x: x["id"]["videoId"],
                        self._persist(lambda: self._search_keyword(self._build_query(song_name, artist_name))),
                    )
                )
            )
        )

    # Returns instances of Search resources
    # https://developers.google.com/youtube/v3/docs/search/list
    def _search_keyword(self, query: str) -> List[Dict]:
        return (
            self.yt_client.search()  # type: ignore
            .list(part="snippet", maxResults=50, q=query, type="video")
            .execute()["items"]
        )

    # returns list of Video resources
    # https://developers.google.com/youtube/v3/docs/videos/list#resource
    def _get_videos_by_id(self, v_ids: List[str]) -> List[dict]:
        # WARN - `id` may expect a comma separated listof video ids
        return (
            self.yt_client.videos()  # type: ignore
            .list(part="snippet,contentDetails,statistics,topicDetails", id=v_ids)
            .execute()["items"]
        )

    def _get_comment_threads(self, video: dict) -> List[dict]:
        return self._persist(
            lambda: self.yt_client.commentThreads()  # type: ignore
            .list(part="snippet,replies", videoId=video["id"])
            .execute()["items"],
        )

    def _update_replies(self, cthread: dict) -> dict:
        comment = cthread["snippet"]["topLevelComment"]
        comment["n_replies"] = cthread["snippet"]["totalReplyCount"]
        try:
            comment["replies"] = cthread["replies"]["comments"]
        except KeyError:
            comment["replies"] = []
        return comment

    def _get_comments(self, video: dict) -> List[Dict]:
        # Convert CommentThread resource to Comment resource with nested Comment list
        return list(map(self._update_replies, self._get_comment_threads(video)))

    def _persist(self, func: Callable[[], List[dict]], retries: int = 3):
        for _ in range(0, retries):
            try:
                return func()
            except HttpError as e:
                if e.status_code == 403:
                    if e.error_details[0]["reason"] == "commentsDisabled":  # type: ignore
                        return []
                    else:
                        print(e)
                        print("Entering 24hr sleep loop")
                        time.sleep(86400)
                        continue

                elif e.status_code == 500:
                    print("500 - Internal Server Error \n")
                    print("Sleeping for 1 hour\n")
                    time.sleep(3600)
                    continue

                elif e.status_code == 400:
                    if e.error_details[0]["reason"] == "missingRequiredParameter":  # type: ignore
                        return []
                    else:
                        print(e.status_code)
                        print(func)
                        # raise e
                        return []
                else:
                    print(e.status_code)
                    raise (e)
        print("Exiting without handling!")
        exit()
