import tweepy
import requests
import praw

from abc import abstractmethod
from typing import List, Callable, TypeVar, Union
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language
from bson.objectid import ObjectId
from googleapiclient.errors import HttpError
from database.driver import Driver
from googleapiclient.discovery import Resource as YoutubeResource

Client = TypeVar("Client", tweepy.Client, praw.Reddit, YoutubeResource)
Error = Union[requests.exceptions.ConnectionError, HttpError]


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

    def _persist(self, func: Callable, exceptions: tuple, retries: int = 3):
        for i in range(0, retries):
            try:
                return func()
            #  except requests.exceptions.ConnectionError:
            except exceptions as e:
                self._handler(e)
        exit()

    @abstractmethod
    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        pass

    @abstractmethod
    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        pass

    @abstractmethod
    def _authenticate(self) -> Client:
        pass

    @abstractmethod
    def _handler(self, e: Error) -> None:
        pass
