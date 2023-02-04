import re
from itertools import chain
from typing import Callable, List

import requests
from bs4 import BeautifulSoup
from bson.objectid import ObjectId
from soundcloud import SoundCloud, Track

from data_mining.commentminer import CommentMiner
from database.driver import Driver

SOUNDCLOUD_BASE_URL = "https://soundcloud.com"


class AuthenticationException(Exception):
    pass


class SoundCloudBot(CommentMiner):
    def __init__(self, _: str):
        self.client = self._authenticate()

    def _authenticate(self) -> SoundCloud:
        homepage = requests.get(SOUNDCLOUD_BASE_URL).content
        homepage_soup = BeautifulSoup(homepage, "html.parser")
        cdn_endpoints = homepage_soup.find_all("script", attrs={"crossorigin": "", "src": re.compile(".*")})

        for endpoint in cdn_endpoints:
            endpoint_js = requests.get(endpoint["src"]).text
            if ',client_id:"' in endpoint_js:
                return SoundCloud(endpoint_js.split(',client_id:"')[1].split('"')[0])
        raise AuthenticationException

    def _persist(self, func: Callable, exceptions: tuple, retries: int = 3):
        pass

    def _get_submissions(self, song_name: str, artist_name: str) -> List[dict]:
        return list(map(dict, self.client.search_tracks(self._build_query(song_name, artist_name))))

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        tracks = self._get_submissions(song["song_name"], song["artist_name"])
        comments = list(chain.from_iterable(map(self._get_comments, tracks)))
        # For every track from List[Track]

        # Get all comments from tracks

        # Map track as dict

        # Map comment as dict

        # Return Track with a list of Comments

        return db.insert_posts(
            comments,
            {
                "artist_name": song["artist_name"],
                "song_name": song["song_name"],
                "dataset": song["Dataset"],
                "source": "Soundcloud",
            },
            {"body": "body"},
            {k: f"$$this.{k}" for k, _ in comments[0].items()},
        )

    def _get_comments(self, track: Track) -> List[dict]:
        return list(map(dict, self.client.get_track_comments(track.id)))
