import argparse
import configparser
import praw
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# We assume all input datasets have a standardized input API
# song_id, valence, arousal, song_name, artist_name
# if your dataset does not match these column headers, please rename them as needed. 

site_name = 'bot1'
header = ["Query Index", "Query", "Song ID", "Valence", "Arousal", "Result Index", "Subreddit", "Subreddit ID",
    "Submission Title", "Submission Body", "Submission ID", "Comment Body", "Comment ID", "Comment Index", "Comment Replies",
    "Comment Score", "Submission Comments", "Submission URL", "Submission Score"]

def main():
    args = parseargs()
    api_key = configparser.ConfigParser()
    api_key.read(args.config)
    reddit = praw.Reddit(site_name,
        client_id = api_key['CLIENT_INFO']['client_id'],
        client_secret = api_key['CLIENT_INFO']['client_secret'])

    dataset = pd.read_csv(args.input)
    path = f"{args.output}/downloads/"
    dispatch_queries(reddit, dataset, path, args.search_depth)

def dispatch_queries(reddit: praw.Reddit, df, path: str, depth: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Beginning mining comments from Reddit...")
    for q_index, row in tqdm(df.iterrows(), total=len(df)):
        dtime = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        csv_fname = f"{path}reddit_{dtime}_{row['song_id']}.csv"
        query = f"title:\"{row['artist_name']}\" \"{row['song_name']} \""
        for p_index, submission in enumerate(get_submissions(query, reddit, depth)):
            song_df = pd.DataFrame(columns=header)
            for c_index, comment in enumerate(get_comments(submission)):
                song_df.append({
                    header[0]: q_index,
                    header[1]: query,
                    header[2]: row['song_id'],
                    header[3]: row['valence'],
                    header[4]: row['arousal'],
                    header[5]: p_index,
                    header[6]: comment.subreddit.display_name,
                    header[7]: comment.subreddit.id,
                    header[8]: submission.title,
                    header[9]: submission.selftext,
                    header[10]: submission.id,
                    header[11]: comment.body,
                    header[12]: comment.id,
                    header[13]: c_index,
                    header[14]: len(comment.replies),
                    header[15]: comment.score,
                    header[16]: submission.num_comments,
                    header[17]: submission.url,
                    header[18]: submission.score
                }, ignore_index=True)
            song_df.to_csv(csv_fname)


def get_submissions(query: str, reddit: praw.Reddit, search_depth: int) -> list:
    subreddit = reddit.subreddit("all")
    return subreddit.search(query, 'top', 'lucene', "all", limit=search_depth)


def get_comments(post):
    post.comments.replace_more(limit=0)
    return list(post.comments)

def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A Reddit bot for gathering social media comments \
        connected to songs. A part of the Music Emotion Prediction project @ OSU-Cascades")
    parser.add_argument('-i', dest='input', required=True, help='Input file. Should be a csv list of songs, \
        containing artist_name and song_title, as well as valence and arousal values.')
    parser.add_argument('-c', dest='config', required=True, help='Config file for PRAW.')
    parser.add_argument('-o', dest='output', required=True, help='Destination folder for output files. Must be a directory.')
    parser.add_argument('--search_depth', dest='search_depth', default=10, type=int,
        help='How many posts the reddit bot should scrape comments from')
    return parser.parse_args()

