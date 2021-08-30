from dataclasses import dataclass, field
from typing import List

@dataclass
class Comment:
    index: int
    id: str
    score: int
    body: str 
    replies: int
    lang: str
    lang_p: float


@dataclass
class Submission:
    index: int
    title: str
    body: str
    lang: str
    lang_p: float
    url: str
    id: str
    score: int
    n_comments: int
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


