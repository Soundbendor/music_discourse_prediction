from dataclasses import dataclass, field
from typing import List

@dataclass
class Comment:
    id: str
    score: int
    body: str 
    reply_count: int
    lang: str
    # TODO - extract this to platform specific data contracts where it is necesscary 
    # reddit(?), youtube(?)
    #  lang_p: float


@dataclass
class Submission:
    title: str
    body: str
    lang: str
    lang_p: float
    url: str
    id: str
    score: int
    n_comments: int
    # TODO - Extract subreddit to reddit specific data contract
    subreddit: str
    comments: List[Comment] = field(default_factory=list)


@dataclass
class SearchResult:
    song_name: str
    artist_name: str
    query_index: int
    dataset: str
    valence: float
    arousal: float
    submissions: List[Submission] = field(default_factory=list)

