from dataclasses import dataclass
from typing import List


from src.data_mining.jsonbuilder import Comment, Submission

# See https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/place
@dataclass
class TwitterContextEntity:
    id: str
    name: str
    description: str


@dataclass
class TwitterContextDomain:
    id: str
    name: str
    description: str

    
@dataclass
class TwitterLocation:
    full_name: str
    id: str
    country_code: str
    # Coordinate info
    geo: dict
    place_type: str


@dataclass
class TwitterComment(Comment):
    cashtags: List[str]
    hashtags: List[str]
    # A list of referenced usernames
    mentions: List[str]
    # URL dict should contain url, display_url, title, and description.
    urls: List[dict]
    # We're just getting the raw contexrt annotations for now, since this data is difficult to process
    context_annotations: List[TwitterContextEntity | TwitterContextDomain]
    author_id: str
    conversation_id: str
    referenced_tweets: List[dict]
    created_at: str
    # place info contained in "includes" - is geo separate?
    places: List[TwitterLocation]
    geo: dict
    retweet_count: int
    entity_annotations: List[dict]


# Problem - Submissions are also tweets, should have same fields.
# In the case of Twitter, n-comments will need to be manually computed after we build the conversation graph
# To avoid multiple inherritence, we inherrit TwitterComment and depend on manually enforcing
#  Submission master format.
@dataclass
class TwitterSubmission(TwitterComment):
    url: str
    # Really represents n_replies 
    n_comments: int
    
    
    
