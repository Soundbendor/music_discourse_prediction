import argparse
import pandas as pd

from datetime import datetime
from .reddit.redditbot import RedditBot
from .youtube.youtubebot import YoutubeBot
from .lyrics.geniusbot import GeniusBot
from .twitter.twitterbot import TwitterBot
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
    # this should probably be lt, not gt, in the future? set to gt for initial load.
    songs = db_client['songs'].find({ 'Dataset': args.dataset.lower(), '$or': [ {'last_modified': {'$gt': args.timestamp }}, {'last_modified': {'$exists': False}}] }, no_cursor_timeout=True)
    
    # cache the cursor to avoid timeout (the timeout parameter is fake)
    songs = [document for document in songs]
    for song in songs:
        print(f"Starting query for song {song['song_name']}")
        db_client['songs'].update_one({'_id': song['_id']}, {"$addToSet": {"Submission": bot.process_submissions(db_client, song)}})
        db_client['songs'].update_one({'_id': song['_id']}, {"$set": {"last_modified": datetime.utcnow()}})


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
    parser.add_argument('--timestamp', dest='timestamp', required=True, type=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), 
                        help="Fetch all songs with a last modified time less than this given UTC timestamp.")
    return parser.parse_args()
