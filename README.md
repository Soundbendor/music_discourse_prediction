# Music Discourse Prediction - SoundBendOR Lab
A tool for predicting the valence and arousal of a song based on social media commentary.

## Modules

Data Mining contains a series of modules used for social media data collection. Currently, we support comment aggregation from Reddit, YouTube, and Twitter. These data scraping bots will output a JSON file for each song in a dataset, containing all submissions and comments which contain a given song title and artist name. We also include an interface to the Genius lyrics API for gathering song lyrics in a JSON format. 

The wordbag module generates a set of statistical features from a music discourse dataset by using one of 5 affective wordlists with valence, arousal, and affective ratings. This returns the minimum, maximum, average, and standard deviation in valence, arousal, and affect from the words contained in a collection of comments assosciated with a given song, for every song in the music affective dataset. It expects an input dataset with the following fields:

|    song_id    | valence | arousal | song_name | artist_name|
|---------------|---------|---------|-----------|------------|
|  ...........  |   ....  |   ....  |    ....   |    ....    |

and will include the true valence and true arousal rating in the output CSV.


The Experiment module performs learning from the wordbag module features to predict musical affect. The parameters for a given experiment are defined in an experiment.ini file. 2 template INI files are inluded in the etc/ diretory. Our prediction tool has two different modes, classification and regression. Classification mode attempts to classify the songs into four unique categories (happy, upset, depressed, calm) and training data is automatically tagged based on the values of it's valence and arousal. A regression experiment will attempt to directly predict the valence and arousal values. Support for testing various models from the sklearn library, parameter tuning using grid search, as well as feature selection and dimensionality reduction using PCA are included. 

Categorization labeling will be performed based on the following:

| arousal_sign | valence_sign | category |
|--------------|--------------|----------|
|       +      |       +      |   happy  |
|       -      |       +      |   upset  |
|       -      |       -      | depressed|
|       +      |       -      |    calm  |


The BERT module includes a direct-learning pipeline, accepting a JSON-formatted set of song comments as an input and directly predicting valence and arousal targets using [distilBERT](https://paperswithcode.com/method/distillbert).


## Notes for Capstone Maintainers
This release is packaged for the purposes of CS-463 Capstone archival and final completion. Please note that this code is not for distribution to any parties besides the current instructors for CS-463 Capstone, as it includes ongoing, unpublished research work. 


## Acknowledgements and Contact
Please cite our paper for any use of this work: [TBD]

Developed by: Aidan Beery

Contact: beerya@oregonstate.edu

Advised by: Dr. Patrick Donnelly

Contact: donnellp@oregonstate.edu

Website: http://www.soundbendor.org/

OSU-Cascades 2019-2022