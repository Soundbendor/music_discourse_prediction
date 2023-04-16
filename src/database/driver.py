import gc
import itertools
from datetime import datetime
from typing import Callable, List, Union

import pandas as pd
import pymongo
from bson.objectid import ObjectId
from more_itertools import chunked
from pymongo.results import InsertManyResult
from tqdm import tqdm


class Driver:
    def __init__(self, db_name: str) -> None:
        self.client = pymongo.MongoClient()[db_name]

    # Returns a list of songs for a given dataset.
    # TODO - Support arbitrary queries (limits, randomized subsets, manual holdout sets)
    # TODO - Change gt to lt after initial load (?)
    def get_dataset(self, ds_name: str, timestamp: datetime) -> List[dict]:
        songs = self.client["songs"].find(
            {
                "Dataset": ds_name,
                "$or": [
                    {"last_modified": {"$lt": timestamp}},
                    {"last_modified": {"$exists": False}},
                ],
            },
            no_cursor_timeout=True,
        )
        return [document for document in songs]

    def _update_reply(self, reply: dict, doc: dict) -> dict:
        # print(doc)
        doc.update(reply)
        return doc

    def _make_replies(self, replies: List, doc: dict) -> List[dict]:
        return [self._update_reply(reply, doc.copy()) for reply in replies]

    def _process_song(self, song: dict, source_type: Union[str, List[str], None]) -> List[dict]:
        ids = list(itertools.chain.from_iterable(song["Submission"]))
        submissions = list(
            [x for x in self.client["posts"].find({"_id": {"$in": ids}, **self._make_source_filter(source_type)})]
        )
        return list(map(lambda x: x | song, submissions))

    def get_discourse(
        self, ds_name: Union[str, List[str], None] = "", source_type: Union[str, List[str], None] = ""
    ) -> pd.DataFrame:
        print("Getting discourse...")
        songs = [x for x in self.client["songs"].find(self._make_dataset_filter(ds_name))]

        # Memory mitigation
        songs1 = songs[: len(songs) // 2]
        songs2 = songs[len(songs) // 2 :]
        dfs = []
        for songs in [songs1, songs2]:
            print("Fetching comments...")
            posts = list(itertools.chain.from_iterable(map(lambda x: self._process_song(x, source_type), tqdm(songs))))
            print("Fetching replies...")
            replies = list(
                itertools.chain.from_iterable(map(lambda x: self._make_replies(x["replies"], x), tqdm(posts)))
            )
            df = pd.DataFrame.from_records(posts + replies)
            dfs.append(df)
            del posts
            del replies
            gc.collect()
        df = pd.concat(dfs, axis=0)
        df = df[["_id", "song_name", "artist_name", "body", "score", "valence", "arousal"]]
        print(df)
        # df.to_csv("nathan_deezer.csv")
        df["source"] = source_type
        return df  # type: ignore

    def new_get_dataset(self, ds_name: str, src_name: str) -> List[dict]:
        retrieved_songs = self.client["posts"].find({"dataset": ds_name, "source": src_name}).distinct("song_name")
        retrieved_songs = [doc for doc in retrieved_songs]
        new_songs = self.client["songs"].find({"Dataset": ds_name, "song_name": {"$nin": retrieved_songs}})
        dd = [doc for doc in new_songs]
        print(len(dd))
        return dd

    # In the case of an empty dataset name string, we want all songs from all datasets.
    def _make_dataset_filter(self, ds_name: Union[str, List[str], None]) -> dict:
        if isinstance(ds_name, str):
            return {"Dataset": ds_name}
        if isinstance(ds_name, List):
            return {"Dataset": {"$in": ds_name}}
        return {}

    # TODO - Add score filtering here
    def _make_source_filter(self, source_type: Union[str, List[str], None]) -> dict:
        if isinstance(source_type, str):
            return {"source": source_type, "score": {"$gt": 1}}
        if isinstance(source_type, List):
            return {"source": {"$in": source_type}}
        return {}

    # Inserts the comment IDs returned from a CommentMiner instance into that song's db entry
    def update_song(self, song: dict, ids: List[ObjectId]) -> None:
        self.client["songs"].update_one({"_id": song["_id"]}, {"$addToSet": {"Submission": ids}})
        self.client["songs"].update_one(
            {"_id": song["_id"]}, {"$set": {"last_modified": datetime.utcnow()}}
        )  # type: ignore

    def update_replies(self, id: ObjectId, reply_ids: List[ObjectId]) -> None:
        self.client["posts"].update_one({"_id": id}, {"$set": {"replies": reply_ids}})

    # Protects database from update_many calls with empty data lists.
    def _make_transaction(self, func: Callable[[List], InsertManyResult], data: List) -> Union[InsertManyResult, None]:
        if data:
            return func(data)
        return None

    # Insert the result from a CommentMiner's API calls into the Posts collection
    # Updates the resulting reponse dict to coerce fields accordint to custom mapping
    # Accepts a mapping of fields to rename, a dataset name, and a list of posts
    # Returns a list of Post IDs
    # TODO: Is dict appropriate type for posts list
    def insert_posts(self, posts: List[dict], metadata: dict, mapping: dict, array_mapping: dict) -> List[ObjectId]:
        insert_response = self._make_transaction(self.client["posts"].insert_many, posts)
        if insert_response:
            self.client["posts"].update_many(
                {"_id": {"$in": insert_response.inserted_ids}},
                {"$set": metadata, "$rename": mapping},
            )
            self.client["posts"].update_many(
                {"_id": {"$in": insert_response.inserted_ids}},
                [{"$set": {"replies": {"$map": {"input": "$replies", "in": array_mapping}}}}],
            )
            return insert_response.inserted_ids
        return []
