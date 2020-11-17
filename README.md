# Speech Recognition model

This is a Deep Learning model that gets an audio input and tries to predict/write down what it is said in the audio.
It is written in Python with Pytorch framework and it is trained in Google Colab.

Code here is uploaded as a .ipynb Notebook, so you can download and open it Google Colaboratory directly.

We have done some experiments to test if our hypothesis is true and also to see how the model performs with different parameters.  

## Hypothesis 

The speech recognition model will generate much better predictions if we add Attention and are able to train with Attention.

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

There is an interesting result that shows how the model without attention repeats sentences and words. It does not care about what it is said in the audio, just outputs the sentence with highest probability. Here you can see some outputs for the model without attention:

![Attention WER](/images/repeat.png)

### Experiment 2: Number of epochs

In this experiment we have trained our attention model with different number of epochs to see how this impacts to the perplexity value. 
Here are the hyperparameter values and the results:

 | Variable | Description | Value |
 | -- | -- | -- |
 | embedding_size | Embedding size | 125 |
 | batch_size | Batch size | 10-12 |
 | learning_rate | Learning rate | 1x10^-4 |
 | num_epochs | Number of epochs | variable |

![Attention batch size 12](/images/att_6k_8ep_emb125_bat12_perplexity.JPG)

![Attention batch size 10](/images/att_6k_8ep_emb125_perplexity.JPG)

We can see through this images about perplexity that it reaches its minimum value when we train 7-8 epochs. Then perplexity would continue decreasing, but very slowly. 


### Experiment 3: Embedding size

This experiment is about changing the embedding size to the attention model. We can see WER for embedding sizes 50, 125 and 175, respectively.

![Attention WER 50](/images/att_6k_8ep_wer.JPG)

![Attention WER 125](/images/att_6k_8ep_emb125_wer.JPG)

![Attention WER 175](/images/att_6k_10ep_emb175_wer.JPG)

It can be seen that WER is smaller for embedding size equal to 125. So results get a little better when we increase the embedding but then get worse if we increase it a little more. 


### Experiment 4: Learning rate

In this experiment we have changed the learning rate to see how it affects the WER results in the model with attention. The results with learning rate 0.0001 and 0.0002 respectively can be seen here:

![Attention lr 0.0001](/images/att_6k_8ep_wer.JPG)

![Attention WER 0.0002](/images/lr0002.png)

We can see how WER gets the best with learning rate equal to 0.0002.

Parameters for this experiment have been:

 | Variable | Description | Value |
 | -- | -- | -- |
 | embedding_size | Embedding size | 125 |
 | batch_size | Batch size | 10 |
 | learning_rate | Learning rate | variable |
 | num_epochs | Number of epochs | 8 |

 With both learning rate values (0.0001 and 0.0002), perplexity goes down and reaches its minimum arround epoch 7-8.
 But, if we now change the learning rate to 0.001, perplexity behaves different:

![Attention training](/images/perplexity_up.png)

We can see how it reaches its minimum at epoch 6 and starts going up again if we continue the training with more epochs.

## General results

If we look at the next images showing some examples of predictions for the model with attention we can see some interesting thigs:

![Attention training](/images/att_6k_8ep_1v.JPG)

We see how the model learns to begin the sentences with a capital letter. It also learns that only one space is placed between words. 

If we look now a prediction of the attention model we can see this:

![Attention prediction](/images/att_6k_8ep_emb125_bat12_pred_example.JPG)

Although the model is not able to predict what it is said in the audio, 
it has learned the spelling of the words (it predicts real words), and inserts them in a logical order (has learned morphosyntactic skills). 

## Setbacks

Due to time and GPU restrictions we have had to reduce our dataset from 40000 audio files and sentences to 6000, so that the model could finish training. If we try to train with more than, aproximately, 6000 files, Google Colab's GPU goes out of RAM.  

## Conclusions


## Future considerations

If we had to improve this model in Google Colab it would be great to consider training in two separate phases: one that would encode our audio files with PASE and then save the resulting tensors, so that we could free almost 9 GB of GPU RAM, and another where we would feed the remaining part of the model and train the parameters with the whole RAM. This would allow us to use more data. 
