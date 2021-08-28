import argparse
import configparser
import dataclasses
import os
import json
import pandas as pd

from .jsonbuilder import SearchResult
from .commentminer import CommentMiner
from .reddit.redditbot import RedditBot
from .youtube.youtubebot import YoutubeBot
from datetime import datetime
from tqdm import tqdm

# We assume all input datasets have a standardized input API
# song_id, valence, arousal, song_name, artist_name
# if your dataset does not match these column headers, please rename them as needed. 


def main():
    args = parseargs()
    bot = args.bot_type(args.config)
    dataset = pd.read_csv(args.input)
    path = f"{args.output}/downloads/"
    dispatch_queries(bot, dataset, path, args.dataset)

def dispatch_queries(miner: CommentMiner, df, path: str, ds_name: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Beginning mining comments from Reddit...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dtime = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        fname = f"{path}reddit_{dtime}_{row['song_id']}.json"
        with open(fname, 'w') as out:
            json.dump(dataclasses.asdict(SearchResult(
                song_name = row['song_name'],
                artist_name = row['artist_name'],
                query_index = idx,
                dataset = ds_name,
                valence = row['valence'],
                arousal= row['arousal'],
                submissions = miner.query(row['song_name'], row['artist_name']))), out, indent=4)

def minertype(istr: str):
    istr = istr.strip().lower()
    choices = {
        'reddit': RedditBot,
        'youtube': YoutubeBot
    }
    try:
        return choices[istr]
    except KeyError:
        raise argparse.ArgumentTypeError('Invalid type option')
        

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Reddit bot for gathering social media comments \
        connected to songs. A part of the Music Emotion Prediction project @ OSU-Cascades")
    parser.add_argument('-i', dest='input', required=True, help='Input file. Should be a csv list of songs, \
        containing artist_name and song_title, as well as valence and arousal values.')
    parser.add_argument('-c', dest='config', required=True, help='Config file for PRAW.')
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, 
        help='The name of the dataset.')
    parser.add_argument('-o', dest='output', required=True, help='Destination folder for output files. Must be a directory.')
    parser.add_argument('--search_depth', dest='search_depth', default=10, type=int,
        help='How many posts the reddit bot should scrape comments from')
    parser.add_argument('-t', '--type', dest='bot_type', required=True, type=minertype, default=RedditBot,
        help='Specify which platform to perform queries on. Options include youtube, reddit.')
    return parser.parse_args()





