# ToxBlock

ToxBlock is a Python machine learning application for recognizing toxic language in text. It can potentially be employed for automatically screening text in articles, posts, and comments on social media, digital news, online forums etc. and blocking it or flagging it for further review by human intelligence.

It can predict probabilities for classifying English text into six categories of verbal toxicity: toxic, severe toxic, obscene, threat, insult, and identity hate. 

ToxBlock currently (`version 0.1.2`) uses a bidirectional LSTM recurrent neural network with a word embedding layer (pre-trained with the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings) and has been trained on the Toxic Comment Data Set ([CC0](https://creativecommons.org/share-your-work/public-domain/cc0/) license) provided by Conversation AI / Jigsaw in a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) in 2018.

*Disclaimer: the data set used for training, test examples in this repo, as well as the usage examples presented here contain toxic language that may be considered profane, vulgar, or offensive. If you do not wish to be exposed to toxic language, DO NOT proceed to read any further.*

## Installation

You can install ToxBlock form the Python package index (PyPI) using `pip`:
```python
pip install tox-block
```

## Usage

The methods for prediction are contained in the module `tox_block.prediction`.

Predictions for single strings of text can me made via `tox_block.prediction.make_single_prediction`:

```python
from tox_block.prediction import make_single_prediction

make_single_prediction("I will kill you, you fucking idiot!", rescale=True)
```
It will return a dictionary with the original text and the predicted probabilities for each category of toxicity:
```python
{'text': 'I will kill you, you fucking idiot!',
 'toxic': 0.9998680353164673,
 'severe_toxic': 0.7870364189147949,
 'obscene': 0.9885633587837219,
 'threat': 0.8483908176422119,
 'insult': 0.9883397221565247,
 'identity_hate': 0.1710592657327652}
```

To make bulk predictions for several texts, they can be passed as a list of strings into `tox_block.prediction.make_predictions`:
```python
from tox_block.prediction import make_predictions

make_predictions(["Good morning my friend, I hope you're having a fantastic day!",
                  "I will kill you, you fucking idiot!",
                  "I do strongly disagree with the fascist views of this joke that calls itself a political party."], rescale=True)
```
It will return a dictionary of dictionaries of which each contains the original text and the predicted probabilities for each category of toxicity:
```python
{
0: {'text': "Good morning my friend, I hope you're having a fantastic day!",
  'toxic': 0.05347811430692673,
  'severe_toxic': 0.0006274021579883993,
  'obscene': 0.004466842859983444,
  'threat': 0.009578478522598743,
  'insult': 0.00757843442261219,
  'identity_hate': 0.002106667961925268},
 1: {'text': 'I will kill you, you fucking idiot!',
  'toxic': 0.9998679757118225,
  'severe_toxic': 0.7870362997055054,
  'obscene': 0.9885633587837219,
  'threat': 0.8483908176422119,
  'insult': 0.9883397221565247,
  'identity_hate': 0.171059250831604},
 2: {'text': 'I do strongly disagree with the fascist views of this joke that calls itself a political party.',
  'toxic': 0.026190076023340225,
  'severe_toxic': 7.185135473264381e-05,
  'obscene': 0.0009493605466559529,
  'threat': 0.00012321282702032477,
  'insult': 0.0029190618079155684,
  'identity_hate': 0.0022098885383456945}
}
```
The boolean parameter `rescale` specifies if the predicted probabilities should be min-max-scaled to be on a similar range. If `rescale=False`, the raw probabilities will be returned.

## Training ToxBlock on your own data

In case you want to extend ToxBlock with data that you collected yourself or got from other sources, follow these steps: First, clone this repo so that you have it on your local machine. 

The data should be saved in a CSV file in the same format as the data from the Toxic Comment [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), which means it should have a column called `comment_text`, which contains the texts that will be used for training, and six columns corresponding to the six categories `toxic`, `severe_toxic`,`obscene`, `threat`, `insult`, and `identity_hate`, which should contain a `1` if the text has been labeled as belonging to the respective category, and `0` otherwise. Either save the file under the path `data/merged.csv` or set the file path you want as value for the variable `tox_block.config.config.MERGED_DATA_FILE`.

Download the [GloVe embedding files](http://nlp.stanford.edu/data/glove.6B.zip), unzip them, and place the `glove.6B.50d.txt` file in the `data/` folder. 
You can then start training the model by:
```python
from tox_block.training_pipeline import run_training

run_training()
```
You can specify any training or model parameters you want to change in the configuration `tox_block.config.config`. After retraining the model, you may also want to update the parameters for min-max-scaling in `tox_block.config.config.RESCALE_PROBA`, which are used for rescaling the predicted probabilities if `rescale=True` in the prediction methods.

## Acknowledgements

Thanks a lot to the organizers of the Toxic Comment [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) for providing the data on which this model is trained, as well as to all the participants of the competition who shared their ideas on how to build great classification models.



