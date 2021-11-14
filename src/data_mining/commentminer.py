import configparser

from abc import ABC, abstractmethod
from typing import List
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language

from data_mining.jsonbuilder import Submission


class CommentMiner:


    def l_detect(self, txt: str):
        try:
            return detect_langs(txt)[0]
        except LangDetectException:
            return Language("?", 1.00)

    def _process_api_key(self, f_key: str) -> configparser.ConfigParser:
        api_key = configparser.ConfigParser()
        api_key.read(f_key)
        return api_key

    @abstractmethod
    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        pass