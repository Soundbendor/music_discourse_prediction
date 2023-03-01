import argparse
from datetime import datetime

from database.driver import Driver

from .lyrics.geniusbot import GeniusBot
from .reddit.redditbot import RedditBot
from .soundcloud.soundcloudbot import SoundCloudBot
from .twitter.twitterbot import TwitterBot
from .youtube.youtubebot import YoutubeBot

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
    bot = args.bot_type(args.config)
    driver = Driver("mdp")
    # songs = driver.get_dataset(args.dataset.lower(), args.timestamp)
    songs = driver.new_get_dataset(src_name=args.source, ds_name=args.dataset.lower())

    for song in songs:
        print(f"Starting query for song {song['song_name']}")
        driver.update_song(song, bot.fetch_comments(driver, song))


def minertype(istr: str):
    istr = istr.strip().lower()
    choices = {
        "reddit": RedditBot,
        "youtube": YoutubeBot,
        "lyrics": GeniusBot,
        "twitter": TwitterBot,
        "soundcloud": SoundCloudBot,
    }
    try:
        return choices[istr]
    except KeyError:
        raise argparse.ArgumentTypeError("Invalid type option")


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A data inject tool for the music semantic discourse dataset.")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="The name of the dataset to query songs from. Uses all songs by default.",
    )
    parser.add_argument(
        "-t",
        "--type",
        dest="bot_type",
        required=True,
        type=minertype,
        help="Specify which platform to query <reddit, youtube, twitter>",
    )
    parser.add_argument(
        "--timestamp",
        dest="timestamp",
        required=True,
        type=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
        help="Fetch all songs with a last modified time less than this given UTC timestamp.",
    )
    parser.add_argument(
        "--config",
        dest="config",
        required=False,
        default="",
        type=str,
        help="Config file for bot",
    )
    parser.add_argument("-s", "--source", dest="source", required=True, type=str)
    return parser.parse_args()
