import itertools
# WARN - May not be compatible with future versions. See PEP 649 https://peps.python.org/pep-0649/
from __future__ import annotations
from dataclasses import dataclass
# Maintained for backwards-compatability, technically not required
from typing import List, Union


from src.data_mining.jsonbuilder import Comment 

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

def filter_context(dict) -> Union[TwitterContextDomain, TwitterContextEntity]:
   return {k:v for k,v in dict.items() if k in fields(TwitterContextEntity)}

def form_context_annotations(annotations: List[dict]) -> List[Union[TwitterContextDomain, TwitterContextDomain]]:
    return itertools.chain([[TwitterContextDomain(**filter_context(v)) if k == 'domain' else TwitterContextEntity(**filter_context(v)) for k, v in y.items()] for y in annotations])
    
    
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
    context_annotations: List[Union[TwitterContextEntity, TwitterContextDomain]]
    author_id: str
    conversation_id: str
    referenced_tweets: List[dict]
    created_at: str
    # place info contained in "includes" - is geo separate?
    places: List[TwitterLocation]
    geo: dict
    retweet_count: int
    entity_annotations: List[dict]
    replies: List[TwitterComment]


# Problem - Submissions are also tweets, should have same fields.
# In the case of Twitter, n-comments will need to be manually computed after we build the conversation graph
# To avoid multiple inherritence, we inherrit TwitterComment and depend on manually enforcing
#  Submission master format.
@dataclass
class TwitterSubmission(TwitterComment):
    url: str
    # Really represents n_replies 
    n_comments: int
    
    
    
