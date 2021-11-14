import socket
import lyricsgenius
from lyricsgenius import genius
import requests
import urllib3

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission

from lyricsgenius import Genius
from typing import Iterator, List, Union
from time import sleep


class GeniusBot(CommentMiner):
    def __init__(self, key: str) -> None:
        keys = self._process_api_key(key)
        self.api = Genius(key)

    def get_submissions(self, song_name: str, artist_name: str) -> List:
        song = self.api.search_song(song_name, artist = artist_name) 
        if song:
            return [song]
        return []

    def process_submissions(self, song) -> Submission:
        lyrics = song.lyrics
        lang = self.l_detect(lyrics)
        # Rate limiter
        sleep(5)
        return Submission(
            title = f"{song.artist} - {song.title}",
            body = lyrics,
            lang = lang.lang,
            lang_p = lang.prob,
            url = song.url,
            score = 0,
            n_comments = 0,
            subreddit = "",
            comments= [],
            id = song.url)


