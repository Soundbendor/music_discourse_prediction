import configparser

from abc import ABC, abstractmethod
from typing import List, Iterator
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

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))
    
    def _build_query(self, song_name: str, artist_name: str) -> str:
        return f"\"{artist_name}\" \"{song_name}\""

    @abstractmethod
    def get_submissions(self, song_name: str, artist_name: str) -> Iterator:
        pass

    @abstractmethod
    def process_submissions(self, s_result) -> Submission:
        pass