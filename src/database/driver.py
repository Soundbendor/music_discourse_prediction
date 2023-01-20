import itertools
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
                    {"last_modified": {"$gt": timestamp}},
                    {"last_modified": {"$exists": False}},
                ],
            },
            no_cursor_timeout=True,
        )
        return [document for document in songs]

    def get_discourse(self, ds_name: str = "", source_type: str = "") -> pd.DataFrame:
        songs = [x for x in self.client["songs"].find(self._make_dataset_filter(ds_name))]
        ids = list(itertools.chain.from_iterable(itertools.chain.from_iterable(map(lambda x: x["Submission"], songs))))
        posts = [x for x in self.client["posts"].find({"_id": {"$in": ids}})]
        replies = list(itertools.chain.from_iterable(map(lambda x: x["replies"], posts)))
        print(len(posts))
        print(len(replies))
        print(replies)
        # print(ids)

    # In the case of an empty dataset name string, we want all songs from all datasets.
    def _make_dataset_filter(self, ds_name: str) -> dict:
        if ds_name:
            return {"Dataset": ds_name}
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
