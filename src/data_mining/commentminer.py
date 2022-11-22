import requests

from abc import abstractmethod
from typing import List, Callable, Union
from time import sleep
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language
from pymongo.database import Database
from pymongo.results import InsertManyResult
from bson.objectid import ObjectId


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

    def make_transaction(
        self, func: Callable[[List], InsertManyResult], data: List
    ) -> Union[InsertManyResult, None]:
        if data:
            return func(data)
        return None

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
    def process_submissions(self, db: Database, song: dict) -> List[ObjectId]:
        pass
