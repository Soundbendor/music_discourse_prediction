import argparse
import configparser
import praw

site_name = 'bot1'
def main():
    args = parseargs()
    api_key = configparser.ConfigParser()
    api_key.read(args.config)
    reddit = praw.Reddit('bot1',
        client_id = api_key['CLIENT_INFO']['client_id'],
        client_secret = api_key['CLIENT_INFO']['client_secret'])


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Reddit bot for gathering social media comments connected to songs. A part of the Music Emotion Prediction project @ OSU-Cascades")
    parser.add_argument('-i', dest='input', help='Input file. Should be a csv list of songs, containing artist_name and song_title, as well as valence and arousal values.')
    parser.add_argument('-c', dest='config', help='Config file for PRAW.')
    return parser.parse_args()

