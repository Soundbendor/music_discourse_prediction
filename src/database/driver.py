import itertools
from more_itertools import chunked
from datetime import datetime
from typing import Callable, List, Union

import pandas as pd
import pymongo
from bson.objectid import ObjectId
from pymongo.results import InsertManyResult


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
        doc.update(reply)
        return doc

    def _make_replies(self, replies: List, doc: dict) -> List[dict]:
        return [self._update_reply(reply, doc.copy()) for reply in replies]

    def get_discourse(self, ds_name: str = "", source_type: str = "") -> pd.DataFrame:
        songs = [x for x in self.client["songs"].find(self._make_dataset_filter(ds_name))]
        ids = list(itertools.chain.from_iterable(itertools.chain.from_iterable(map(lambda x: x["Submission"], songs))))

        posts = list(
            itertools.chain.from_iterable(
                [
                    [
                        x
                        for x in self.client["posts"].find(
                            {"_id": {"$in": id_sub}, **self._make_source_filter(source_type)}
                        )
                    ]
                    for id_sub in chunked(ids, 10)
                ]
            )
        )
        replies = list(itertools.chain.from_iterable(map(lambda x: self._make_replies(x["replies"], x), posts)))
        print(replies)
        df = pd.DataFrame.from_records(posts + replies)
        df = df[["_id", "song_name", "artist_name", "body"]]
        print(df)
        # print(ids)

    def new_get_dataset(self, ds_name: str, src_name: str) -> List[dict]:
        retrieved_songs = self.client["posts"].find({"dataset": ds_name, "source": src_name}).distinct("song_name")
        retrieved_songs = [doc for doc in retrieved_songs]
        new_songs = self.client["songs"].find({"Dataset": ds_name, "song_name": {"$nin": retrieved_songs}})
        # print(len([doc for doc in new_songs]))
        return [doc for doc in new_songs]

    # In the case of an empty dataset name string, we want all songs from all datasets.
    def _make_dataset_filter(self, ds_name: str) -> dict:
        if ds_name:
            return {"Dataset": ds_name}
        return {}

    def _make_source_filter(self, source_type: str) -> dict:
        if source_type:
            return {"source": source_type}
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
