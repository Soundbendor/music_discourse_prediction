from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment
from typing import List
import googleapiclient.discovery as gapiclient
import google_auth_oauthlib.flow as gauth



scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


class YoutubeBot(CommentMiner):


    def __init__(self, key: str) -> None:
        yt_client = self._make_client(key)


    def _make_client(self, key: str) -> gapiclient.Resource:
        flow = gauth.InstalledAppFlow.from_client_secrets_file(key, scopes)
        creds = flow.run_console()
        return gapiclient.build('youtube', 'v3', credentials=creds)


    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        return list(map(self.process_submissions, self.get_submissions(song_name, artist_name)))
