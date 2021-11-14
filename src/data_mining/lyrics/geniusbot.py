import socket
import requests
import urllib3

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission

from lyricsgenius import Genius
from typing import List
from time import sleep


class GeniusBot(CommentMiner):
    def __init__(self, key: str) -> None:
        keys = self._process_api_key(key)
        self.api = Genius(key)

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        for _ in range(0,3):
            try:
                song = self.api.search_song(song_name, artist = artist_name)
                if song:
                    lyrics = song.lyrics
                    if lyrics:
                        lang = self.l_detect(lyrics)
                        # Rate limiter
                        sleep(5)
                        return [Submission(
                            title = self._build_query(song_name, artist_name),
                            body = lyrics,
                            lang = lang.lang,
                            lang_p = lang.prob,
                            url = song.url,
                            score = 0,
                            n_comments = 0,
                            subreddit = "",
                            comments= [],
                            id = song.url
                        )]
                return []
            # None of this works evidently but I'm leaving it for posterity
            except (socket.timeout, requests.exceptions.Timeout, requests.exceptions.ReadTimeout, urllib3.exceptions.ReadTimeoutError):
                print("Server timeout error - enter sleep loop")
                sleep(100)
                continue
        raise Exception("Retries failed")


    def _build_query(self, song_name: str, artist_name: str) -> str:
        return f"\"{artist_name}\" \"{song_name}\""

    
