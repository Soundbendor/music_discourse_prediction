import requests
import tweepy
import praw

from abc import abstractmethod
from typing import List, Callable, TypeVar
from time import sleep
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language
from pymongo.database import Database
from bson.objectid import ObjectId
from googleapiclient.discovery import Resource as YoutubeResource

Client = TypeVar("Client", tweepy.Client, praw.Reddit, YoutubeResource)


class CommentMiner:
    def l_detect(self, txt: str):
        try:
            return detect_langs(txt)[0]
        except LangDetectException:
            return Language("?", 1.00)

    def _build_query(self, song_name: str, artist_name: str) -> str:
        return '"{}" "{}"'.format(
            artist_name.replace('"', ""), song_name.replace('"', "")
        )

    def persist(self, func: Callable[[], List], retries: int = 3) -> List:
        conn = 0
        while conn <= retries:
            try:
                return func()
            except requests.exceptions.ConnectionError:
                sleep(5)
                print(f"Connection error! Retry #{conn}")
                retries += 1
        exit()

    @abstractmethod
    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        pass

    @abstractmethod
    def fetch_comments(self, db: Database, song: dict) -> List[ObjectId]:
        pass

    @abstractmethod
    def _authenticate(self) -> Client:
        pass
