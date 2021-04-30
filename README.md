# Music Discourse Prediction
## SoundBendOR Lab
A tool for analyzing and predicting the semantic value of songs based on social media commentary. 

Note - Expects an input of social media commentary features, not raw text. See [music_discourse_features](https://github.com/Soundbendor/music_discourse_features) for how to generate the features needed to input here. Expects CSV files with the following format: 

|...metadata... | existing_valence | existing_arousal | ...features... |
|---------------|------------------|------------------|----------------|
|  sample_data  |    sample_data   |    sample_data   |   sample_data  |

where valence, arousal are the values which were are going to attempt to predict.

This prediction tool has two different modes, classification mode, and regression mode. Classification mode attempts to classify the songs into four unique categories (happy, upset, depressed, calm) and training data is automatically tagged based on the values of it's valence and arousal.

| arousal_sign | valence_sign | category |
|       +      |       +      |   happy  |
|       -      |       +      |   upset  |
|       -      |       -      | depressed|
|       +      |       -      |    calm  |

Please cite our paper for any use of this work: [TBD]
Developed by: Aidan Beery
Advised by: Dr. Patrick Donnelly
OSU-Cascades 2019-2021