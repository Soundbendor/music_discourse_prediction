from time import sleep
from googleapiclient.errors import HttpError
from .youtubeinterface import YoutubeInterface, YoutubeSearchResult
from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment

from typing import Dict, List

base_url = 'http://youtu.be/'

class YoutubeBot(CommentMiner):


    def __init__(self, key: str, search_depth: int = 10) -> None:
        self.yt_client = YoutubeInterface(key)
        self.search_depth = search_depth


    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))



    def get_submissions(self, song_name: str, artist_name: str) -> List[YoutubeSearchResult]:
        for i in range(0,3):
            try:
                return self.yt_client.search_by_keywords(
                    query=self._build_query(song_name, artist_name), limit=self.search_depth
                )
            except HttpError as e:
                print(e)
                print(self._build_query(song_name, artist_name))
                print("Entering 24hr sleep loop")
                sleep(86400)
                continue


    def _build_query(self, song_name: str, artist_name: str) -> str:
        return f"\"{artist_name}\" \"{song_name}\""

    def _get_comment_count(self, v_resource: Dict) -> int:
        try: 
            return int(v_resource['statistics']['commentCount'])
        except KeyError:
            return 0


    def _get_video_score(self, v_resource: Dict) -> int:
        try: 
            return int(v_resource['statistics']['likeCount']) - int(v_resource['statistics']['dislikeCount'])
        except KeyError:
            return 0

    def process_submissions(self, s_result: YoutubeSearchResult) -> Submission:
        s_lang = self.l_detect(f"{s_result.snippet['snippet']['title']} {s_result.snippet['snippet']['description']}")
        for i in range(0,3):
            try:
                return Submission(
                    title = s_result.snippet['snippet']['title'],
                    body = s_result.snippet['snippet']['description'],
                    lang = s_lang.lang,
                    lang_p = s_lang.prob,
                    url = base_url + s_result.snippet['id']['videoId'],
                    id = s_result.snippet['id']['videoId'],
                    score = self._get_video_score(s_result.video),
                    n_comments = self._get_comment_count(s_result.video),
                    subreddit = s_result.video['snippet']['channelTitle'],
                    comments = list(map(self.process_comments, self.yt_client.get_comments(s_result.snippet['id']['videoId'])))
                )
            except HttpError as e:
                print(e)
                print(s_result.snippet['snippet']['title'])
                print("Entering 24hr sleep loop")
                sleep(86400)
                continue


    def process_comments(self, comment: dict) -> Comment:
        c_lang = self.l_detect(comment['snippet']['textOriginal'])
        return Comment(
            id = comment['id'],
            score = int(comment['snippet']['likeCount']),
            body = comment['snippet']['textOriginal'],
            replies = 0,
            lang = c_lang.lang,
            lang_p = c_lang.prob
        )