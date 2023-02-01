from data_mining.commentminer import CommentMiner
import requests
from database.driver import Driver
from bson.objectid import ObjectId
from bs4 import BeautifulSoup
from soundcloud import SoundCloud
from typing import Callable, List
import re

SOUNDCLOUD_BASE_URL = "https://soundcloud.com"


class AuthenticationException(Exception):
    pass


class SoundCloudBot(CommentMiner):
    def __init__(self):
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

    def _get_submissions(self, song_name: str, artist_name: str) -> List:
        pass

    def fetch_comments(self, db: Driver, song: dict) -> List[ObjectId]:
        self.client.search_tracks(self._build_query(song["song_name"], song["artist_name"]))
