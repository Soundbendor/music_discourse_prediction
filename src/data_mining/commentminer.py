from abc import ABC, abstractmethod

from typing import List
from data_mining.jsonbuilder import Submission
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language

class CommentMiner:


    def l_detect(self, txt: str):
        try:
            return detect_langs(txt)[0]
        except LangDetectException:
            return Language("?", 1.00)

    @abstractmethod
    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        pass