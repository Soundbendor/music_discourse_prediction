import argparse
import dataclasses
import os
import json
import pandas as pd

from .jsonbuilder import SearchResult
from .commentminer import CommentMiner
from .reddit.redditbot import RedditBot
from .youtube.youtubebot import YoutubeBot
from .lyrics.geniusbot import GeniusBot
from .twitter.twitterbot import TwitterBot
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient

# We assume all input datasets have a standardized input API
# song_id, valence, arousal, song_name, artist_name
# if your dataset does not match these column headers, please rename them as needed.

# Definitions:
# Post: Some social media entity which is related to a Song.
# Can include:
#   Submissions:
#       - Reddit Submission
#       - YouTube Video
#   Comment
#       - Reddit
#       - YouTube
#       - Twitter

def main():
    args = parseargs()
    bot = args.bot_type()
    db_client = MongoClient()['mdp']
    songs = db_client['songs'].find({'Dataset': args.dataset.lower()})
    
    for song in songs:
        song.update({"$addToSet": {"Submission": bot.process_submissions(db_client, song)}})


#  def dispatch_queries(miner: CommentMiner, df, path: str, ds_name: str):
    #  os.makedirs(os.path.dirname(path), exist_ok=True)
    #  print(f"Beginning mining comments from {miner.__class__.__name__}...")
    #
    #  for idx, row in tqdm(df.iterrows(), total=len(df)):
    #      dtime = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    #      fname = f"{path}{miner.__class__.__name__}_{dtime}_{row['song_id']}.json"
    #      with open(fname, 'w') as out:
    #          json.dump(dataclasses.asdict(SearchResult(
    #              song_name=row['song_name'],
    #              artist_name=row['artist_name'],
    #              query_index=idx,
    #              dataset=ds_name,
    #              valence=row['valence'],
    #              arousal=row['arousal'],
    #              submissions=miner.query(row['song_name'], row['artist_name']))),
    #              out, indent=4, ensure_ascii=False)
#

def minertype(istr: str):
    istr = istr.strip().lower()
    choices = {
        'reddit': RedditBot,
        'youtube': YoutubeBot,
        'lyrics': GeniusBot,
        'twitter': TwitterBot
    }
    try:
        return choices[istr]
    except KeyError:
        raise argparse.ArgumentTypeError('Invalid type option')


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A data inject tool for the music semantic discourse dataset.")
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='The name of the dataset to query songs from. Uses all songs by default.')
    parser.add_argument('-t', '--type', dest='bot_type', required=True, type=minertype, 
                        help='Specify which platform to query <reddit, youtube, twitter.')
    return parser.parse_args()
