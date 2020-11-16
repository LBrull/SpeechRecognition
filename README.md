# Speech Recognition model

This is a Deep Learning model that gets an audio input and predicts/writes down what it is said in the audio.
It is written in Python with Pytorch framework and it is trained in Google Colab.

## Hypothesis 

The speech recognition model will generate much better predictions if we add Attention and are able to train with attention.

## Environment: Google Colab setup

It is possible to run/train this model in Google Colab. In order to do this, you need a google account and Google Drive access. Then you have to follow the next steps:

* First of all, we need to download this audio encoder named PASE from this 	github repo: https://github.com/santi-pdp/pase

* Place PASE (the whole folder named pase-master) in a folder named "Project AI" in the root path of your Google Drive. 

* Now you have to download this file (FE_e199.ckpt) from here https://drive.google.com/file/d/1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW/view and place it inside pase-master folder.

* The dataset can be downloaded from here: 
    * Audios dataset: https://drive.google.com/drive/folders/1oQGKrV5JCZ6EHp4k3UBtqYUreuxensB4?usp=sharing
    * Texts dataset: https://drive.google.com/drive/folders/18MuvbMBwuhl8zs4cb7h4M1o0b3tAYI7O?usp=sharing
    * It can be downloaded alternatively from here: https://groups.csail.mit.edu/sls/downloads/flickraudio/index.cgi

* Both flickr_text and flickr_audio datasets need to be placed in "Project AI" folder.

* Finally we only need to select "type of environment" -> GPU at Google Colab and run all the "Install & Import" section instructions.

## The dataset

We have used a Flickr dataset that consists of 40000 audio files and 40000 text files. In the text files you can read what it is said in the audio files. 

## Data preprocessing

In order to train the model, text dataset needs to be preprocessed. We have done the next modifications to all the sentences in order to create the dictionary:

* We have removed the capital letters
* We have removed accent marks
* We have removed punctuation marks

Before feeding our model and begin training it we have ensured that lemmatization is done to all of the sentences. 

## Experiments

### Experiment 1: Attention vs No Attention

We have defined two models, one with attention and one without it, and trained both. Training in order to develop this experiment has been done using the same hyperparameters (embedding size, batch size, learning rate, and number of epochs) in order to test only the impact of adding attention.

 | Variable | Description | Value |
 | -- | -- | -- |
 | embedding_size | Embedding size | 50 |
 | batch_size | Batch size | 10 |
 | learning_rate | Learning rate | 1x10^-4 |
 | num_epochs | Number of epochs | 8 |

We can see here the results when we calculate WER (Word Error Rate) on the model without attention: 

![No attention WER](/images/no_att_6k_8ep_wer.JPG)

And here the WER data about the model with attention:

![Attention WER](/images/att_6k_8ep_wer.JPG)

### Experiment 2: Number of epochs

In this experiment we have trained our attention model with different number of epochs to see how this impacts to the final WER results and the final loss value. Here are the hyperparameter values and the results

 | Variable | Description | Value |
 | -- | -- | -- |
 | embedding_size | Embedding size | 50 |
 | batch_size | Batch size | 10 |
 | learning_rate | Learning rate | 1x10^-4 |
 | num_epochs | Number of epochs | variable |

 * num_epochs = 3

 * num_epochs = 5

 * num_epochs = 8

 * num_epochs = 10

 * num_epochs = 12


### Experiment 3: Embedding size

This experiment is about changing the embedding size to the attention model. We can see WER for embedding sizes 50, 125 and 175, respectively.

![Attention WER 50](/images/att_6k_8ep_wer.JPG)

![Attention WER 125](/images/att_6k_8ep_emb125_wer.JPG)

![Attention WER 175](/images/att_6k_10ep_emb175_wer.JPG)

Here we can see how does perplexity value change with every epoch if we change embedding size from 125 to 175:

![Attention perplexity 125](/images/att_6k_8ep_emb125_perplexity.JPG)

![Attention perplexity 175](/images/att_6k_10ep_emb175_perplexity.JPG)

## Problems 


## Conclusions
