import tweepy
import praw

from abc import abstractmethod
from typing import List, Callable, TypeVar
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language
from bson.objectid import ObjectId
from database.driver import Driver
from googleapiclient.discovery import Resource as YoutubeResource
from soundcloud import SoundCloud

Client = TypeVar("Client", SoundCloud, tweepy.Client, praw.Reddit, YoutubeResource)


class CommentMiner:
    def __init__(self, config: str) -> None:
        pass

    def l_detect(self, txt: str):
        try:
            return detect_langs(txt)[0]
        except LangDetectException:
            return Language("?", 1.00)

    def _build_query(self, song_name: str, artist_name: str) -> str:
        return '"{}" "{}"'.format(artist_name.replace('"', ""), song_name.replace('"', ""))

    @abstractmethod
    def _persist(self, func: Callable, exceptions: tuple, retries: int = 3):
        pass

    @abstractmethod
    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        pass

    @abstractmethod
    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        pass

    @abstractmethod
    def _authenticate(self) -> Client:
        pass
