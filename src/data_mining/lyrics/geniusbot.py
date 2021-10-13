from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment

from lyricsgenius import Genius
from typing import List


class GeniusBot(CommentMiner):
    def __init__(self, key: str, search_depth: int = 10) -> None:
        self.api = Genius(key)

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        song = self.api.search_song(song_name, artist = artist_name)
        
