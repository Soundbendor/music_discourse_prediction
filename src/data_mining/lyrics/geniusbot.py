import argparse
import pandas as pd

from data_mining.commentminer import CommentMiner
from data_mining.jsonbuilder import Submission, Comment

from lyricsgenius import Genius
from typing import List
from tqdm import tqdm


class GeniusBot(CommentMiner):
    def __init__(self, key: str) -> None:
        self.api = Genius(key)

    def query(self, song_name: str, artist_name: str) -> List[Submission]:
        song = self.api.search_song(song_name, artist = artist_name)
        song.save_lyrics()
    


def main():
    args = parseargs()
    df: pd.DataFrame = pd.read_csv(args.input)
    path = f"{args.output}/downloads/"
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pass


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Reddit bot for gathering social media comments \
        connected to songs. A part of the Music Emotion Prediction project @ OSU-Cascades")
    parser.add_argument('-i', dest='input', required=True, help='Input file. Should be a csv list of songs, \
        containing artist_name and song_title, as well as valence and arousal values.')
    parser.add_argument('-c', dest='config', required=True, help='API key for Genius.')
    parser.add_argument('-o', dest='output', required=True, help='Destination folder for output files. Must be a directory.')

