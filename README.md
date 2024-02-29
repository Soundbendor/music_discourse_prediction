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

# Installation

To install this package
1. Use a Python package management solution (e.g. [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), [conda](https://docs.conda.io/en/latest/) to create an environment from the `environment.yml` file.
   `conda env create --name mdp --file environment.yml`
2. Activate your new environment.
   `conda activate mdp`
3. Install the `music_discourse_prediction` package to your local environment
   `pip install .`

# Usage

## Data Mining

## Model Training

To train a new model, use the `bert_features` command. 

Commands:

`[--dataset]`: The name of the dataset which the songs come from. Required. Options are AMG1608, DEAM, PmEmo, or Deezer.

`[--source]`: List of social media sources from which to use comments from. Required. Options are [Youtube, Reddit, Twitter]. 

`[--epochs]`: Number of epochs to fine-tune for. Optional. Default is 2.

`[--batch_size]`: Batch size per GPU. Required. Default is 16.

`[--length]`: Filter command. Drop all comments below a certain number of characters. Optional. Default is 32.

`[--score]`: Filter command. Drop all comments below a certain number of likes. Optional. Default is 3.

`[--model_name]`: HuggingFace model name of a BERT-like model. Default: `distilbert-base-cased`.

`[--input_dir]`: Path to a valid `.csv` which contains a dataset of music discourse comments. Optional. Used in place of a MongoDB instance running on `localhost`.

### Setting Up A Dataset
We provide two options for attaching a dataset to our model training API. The model API will, by default, search for MongoDB instance running on `localhost`. If an input CSV is provided, the model will use that training dataset instead, and an active MongoDB server will not be required to run the model.

#### Option 1: MongoDB

Without any arguments, `bert_features` will default to pulling a social media dataset from a locally hosted MongoDB instance. The program attempt to connect to MongoDB over its default port (27017). We provide a MongoDB image containing our dataset [here]. This database contains two collections, `songs` and `posts`. The `songs` collection contains ~20,000 records of songs labeled for valence and arousal from the [AMG1608](https://ieeexplore.ieee.org/document/7178058), [PmEmo](https://github.com/HuiZhangDB/PMEmo), [DEAM](https://cvml.unige.ch/databases/DEAM/) and [Deezer2018](https://research.deezer.com/publication/2018/09/26/ismir-delbouys.html) datasets. The `posts` collection contains 20,000,000+ social media comments from Reddit, YouTube, and Twitter which mention any of the songs from the four music emotion recognition datasets used in our study. 

You can select subsets of our dataset, slicing by MER dataset using the `--dataset` flag, and by social media source (Twitter, Youtube, Reddit) with the `--source` flag.


#### Option 2: 

`bert_features` also accepts datasets in a `.csv` format. You can generate a `.csv` dataset from the above MongoDB instance using [mongoexport](https://www.mongodb.com/docs/database-tools/mongoexport/) By using a .csv input, you can save preprocessing time, as no query to the database server will be needed in order to retrieve the model training data. This can be useful in deployments where it is difficult to run a database server concurrently with the model training (e.g. HPC clusters). You can retrieve a subset of the dataset by filtering for songs from a specific MER dataset (AMG1608, DEAM, Deezer, PmEmo), or posts from a specific social media source (Twitter, YouTube, Reddit). Once you have this CSV, you can provide it to the model with `--input`. If no `--input` command is provided, the model will assume a database is in use and attempt to connect to it over localhost. 


## Acknowledgements and Contact
If you would like to use our code or dataset, please cite our publication here:

```
@inbook{
   Beery_Donnelly_2024,
   title={Learning Affective Responses to Music from Social Media Discourse}, 
   ISBN={978-3-031-44260-5},
   url={https://doi.org/10.1007/978-3-031-44260-5_6},
   DOI={10.1007/978-3-031-44260-5_6},
   booktitle={Practical Solutions for Diverse Real-World NLP Applications},
   publisher={Springer International Publishing}, 
   author={Beery, Aidan and Donnelly, Patrick J.}, 
   year={2024}, 
   pages={93–119}
}
```

Developed by: Aidan Beery - mail: beerya@oregonstate.edu

Advised by: Dr. Patrick J. Donnelly - mail: donnellp@oregonstate.edu

Website: http://www.soundbendor.org/
