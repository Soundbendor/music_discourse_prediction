import pymongo
from datetime import datetime
from typing import Callable, List, Union
from pymongo.results import InsertManyResult
from bson.objectid import ObjectId


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

    # Inserts the comment IDs returned from a CommentMiner instance into that song's db entry
    def update_song(self, song: dict, ids: List[ObjectId]) -> None:
        self.client["songs"].update_one(
            {"_id": song["_id"]}, {"$addToSet": {"Submission": ids}}
        )
        self.client["songs"].update_one(
            {"_id": song["_id"]}, {"$set": {"last_modified": datetime.utcnow()}}
        )  # type: ignore

    def update_replies(self, id: ObjectId, reply_ids: List[ObjectId]) -> None:
        self.client["posts"].update_one({"_id": id}, {"$set": {"replies": reply_ids}})

    # Protects database from update_many calls with empty data lists.
    def _make_transaction(
        self, func: Callable[[List], InsertManyResult], data: List
    ) -> Union[InsertManyResult, None]:
        if data:
            return func(data)
        return None

    # Insert the results from a CommentMiner's API calls into the Posts collection
    # Updates the resulting reponse dict to coerce fields accordint to custom mapping
    # Accepts a mapping of fields to rename, a dataset name, and a list of posts
    # Returns a list of Post IDs
    # TODO: Is dict appropriate type for posts list
    def insert_posts(
        self, posts: List[dict], metadata: dict, mapping: dict
    ) -> List[ObjectId]:
        insert_response = self._make_transaction(
            self.client["posts"].insert_many, posts
        )
        if insert_response:
            self.client["posts"].update_many(
                {"_id": {"$in": insert_response.inserted_ids}},
                {"$set": metadata, "$rename": mapping},
            )
            return insert_response.inserted_ids
        return []
