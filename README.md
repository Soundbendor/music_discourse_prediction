# Music Discourse Prediction - soundbendor lab

A package for training music emotion recognition models using social media comments. Predicting continous music emotion labels (e.g. valence, arousal) is limited by 

You can find our paper here: https://link.springer.com/chapter/10.1007/978-3-031-44260-5_6 

(full txt: https://aidan-b1409.github.io/files/music_emotion_sact.pdf)

# Installation

To install this package
1. Use a Python package management solution (e.g. [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), [conda](https://docs.conda.io/en/latest/) to create an environment from the `environment.yml` file.
   `conda env create --name mdp --file environment.yml`
2. Activate your new environment.
   `conda activate mdp`
3. Install the `music_discourse_prediction` package to your local environment
   `pip install .`
[Opt]. Install the MongoDB server from [here](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/) in order to use our data collection pipeline. 

# Usage

Our work consists of two main contributions, represented by the `data_mining` module used for social media musical discourse data collection, and the `bert_features` module, which is used for training and evaluating BERT-like pretrained large language models for the task of predicting music valence and arousal targets from only social media comments.

## Data Mining

Our data collection approach depends on a MongoDB instance running on `localhost`, on the default port 27017. The `data_mining` module connects to this local database, and initializes an API connection to the specified social media service. From this, we form a search query to the social media API strictly including the song title and artist name, and return some subset of the top submissions. For Reddit this is all submissions returned by the search API, for Twitter it is the top 100 tweets, and for YouTube it is the top 50 videos. For each of these top-level submissions, we then pull all reply comments or tweets in response to that original post which explicitly mentions the song title and artist name. Each top-level contribution and reply is stored as a separate record in the `posts` collection

Arguments:

`[--dataset]`: The dataset from which to pull query songs. Options: deam, amg1608, deezer, pmemo

`[--type]`: Which social media source to query from. Options: youtube, reddit, twitter

`[--timestamp]`: The last known 'good' timestamp. If there was an error that resulted in a crash during your data pull, you can query the database instance and find the timestamp of the last successful pull. The bot will resume at the song it left off at. 

`[--config]` The config file for the scraping bot. For the Reddit bot, this is in the form of a `praw.ini` fle. 


### Initializing the Database

In the `datasets` folder, we provide four datasets of musical samples annotated for valence and arousal: [AMG1608](https://ieeexplore.ieee.org/document/7178058), [PmEmo](https://github.com/HuiZhangDB/PMEmo), [DEAM](https://cvml.unige.ch/databases/DEAM/) and [Deezer2018](https://research.deezer.com/publication/2018/09/26/ismir-delbouys.html). Our data scraping workflow depends on these datasets being loaded into your MongoDB instance. The `mongo_songs` command will allow you to quickly load and insert these datasets into your database instance. This script will read the CSV and insert each song, with it's assosciated valence and arousal label, into the `songs` collection. 

Arguments:
`[--input]`: The path of the csv containing the annotated samples. 

Note: For the Deezer2018 dataset, the authors define explicit train, test, and validation splits. We retain these splits in separate csv files. However, our data loading script looks for all three files: `deezer_train.csv`, `deezer_test.csv`, and `deezer_validation.csv` if any one of them is supplied from `--input` So, if you run `mongo_songs --input datasets/DEEZER_test.csv`, it will load all of the songs from all three Deezer dataset files. So, running the load command for all three files is unnecesscary. 

## Model Training

To train a new model, use the `bert_features` command. 

Arguments:

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
