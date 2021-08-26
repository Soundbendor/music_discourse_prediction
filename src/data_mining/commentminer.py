from abc import ABC, abstractmethod

from typing import List
from data_mining.jsonbuilder import Submission


class CommentMiner:

    @abstractmethod
    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        pass