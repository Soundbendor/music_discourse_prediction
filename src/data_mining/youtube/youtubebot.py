import time
import os
from google.oauth2.credentials import Credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from time import sleep
from googleapiclient.errors import HttpError
from itertools import chain
from bson.objectid import ObjectId
from googleapiclient.discovery import Resource
from data_mining.commentminer import CommentMiner, Error
from typing import Callable, Dict, List, cast
from database.driver import Driver

BASE_URL = "http://youtu.be/"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
ERR = (HttpError,)


class YoutubeBot(CommentMiner):
    def __init__(self, key: str) -> None:
        self.yt_client = self._authenticate(key)

    def _authenticate(self, f_key: str) -> Resource:
        if os.path.exists("yt_token.json"):
            creds = Credentials("yt_token.json")
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                f_key, SCOPES
            )
            creds = flow.run_local_server(port=0)
            with open("yt_token.json", "w") as token:
                token.write(creds.to_json())
        return googleapiclient.discovery.build(
            API_SERVICE_NAME, API_VERSION, credentials=creds
        )

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:

        videos = self._get_submissions(song["artist_name"], song["song_name"])
        comments = list(chain.from_iterable(map(self._get_comments, videos)))
        print(comments)

        #  s_lang = self.l_detect(
        #      f"{s_result.snippet['snippet']['title']} {s_result.snippet['snippet']['description']}"
        #  )
        #  s_lang = self.l_detect()
        #  return Submission(
        #      title=s_result.snippet["snippet"]["title"],
        #      body=s_result.snippet["snippet"]["description"],
        #      lang=s_lang.lang,
        #      lang_p=s_lang.prob,
        #      url=BASE_URL + s_result.snippet["id"]["videoId"],
        #      id=s_result.snippet["id"]["videoId"],
        #      score=self._get_video_score(s_result.video),
        #      n_comments=self._get_comment_count(s_result.video),
        #      subreddit=s_result.video["snippet"]["channelTitle"],
        #      comments=list(
        #          map(
        #              self.process_comments,
        #              self.yt_client.get_comments(s_result.snippet["id"]["videoId"]),
        #          )
        #      ),

    #
    #  )

    # Returns a list of Video objects
    def _get_submissions(self, song_name: str, artist_name: str) -> List[dict]:
        return self._persist(
            lambda: self._get_videos_by_id(
                list(
                    map(
                        lambda x: x["id"]["videoId"],
                        self._persist(
                            lambda: self._search_keyword(
                                self._build_query(song_name, artist_name)
                            ),
                            ERR,
                        ),
                    )
                )
            ),
            ERR,
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
            lambda: self.yt_client.commentThreads().list(  # type: ignore
                part="snippet,replies", videoId=video["id"]
            ),
            ERR,
        ).execute()["items"]

    def _get_comments(self, video: dict) -> List[Dict]:
        return list(
            chain.from_iterable(
                map(self._flatten_threads, self._get_comment_threads(video))
            )
        )

        #  c_lang = self.l_detect(comment["snippet"]["textOriginal"])

    #          return Comment(
    #      id=comment["id"],
    #      score=int(comment["snippet"]["likeCount"]),
    #      body=comment["snippet"]["textOriginal"],
    #      replies=0,
    #      lang=c_lang.lang,
    #      lang_p=c_lang.prob,
    #  )
    #
    def _update_reply_depth(self, comment: dict) -> dict:
        comment["snippet"]["depth"] = 1
        return comment

    def _flatten_threads(self, thread: dict) -> List[Dict]:
        thread["snippet"]["topLevelComment"]["depth"] = 0
        try:
            replies = list(map(self._update_reply_depth, thread["replies"]["comments"]))
            replies.insert(0, thread["snippet"]["topLevelComment"])
            return thread["replies"]["comments"]
        except KeyError:
            return [thread["snippet"]["topLevelComment"]]

    def _handler(self, e: Error) -> None:
        if type(e) == HttpError:
            # At this point we have confirmed the execption is an HttpError.
            # Provide inference to the type checker
            e = cast(HttpError, e)
            if e.status_code == 403:
                print(e)
                print("Entering 24hr sleep loop")
                time.sleep(86400)
            elif e.status_code == 500:
                print("500 - Internal Server Error \n")
                print("Sleeping for 1 hour\n")
                time.sleep(3600)
            else:
                print(e.status_code)
                raise (e)
