# BERT Hierarchical Attention Network
Hierarchical-Attention-Network for Document Classification implementation in PyTorch with a replacement of the traditional BiLSTM with BERT model.

This repository is an implementation of the article [Hierarchical Attention Networks for Document Classification
](https://www.aclweb.org/anthology/N16-1174/) (Yang et al.) such that one can choose if to use a
traditional BiLSTM for creating sentence embeddings for each sentence or to use BERT for this task (configurable).
If one chooses to use BERT in order to create sentence embedding for each sentence, then the rest of the network
architecture is the same like in the original paper, i.e. feeding the sentence embeddings into BiLSTM encoder with attention
to get a fixed length document vector, that in turn, is fed into a Multi Layer Perceptron with a Softmax activation
aligned with the number of different classes of the chosen data set.


**Original architecture**
![han](./images/han.jpg)

## Setup Instructions
Install `pipenv` with the following command:

```
$ pip install pipenv
```

Open pipenv environment in a new shell:

```
$ pipenv shell
```

Add the project to PYTHONPATH:

```
$ export PYTHONPATH=$PYTHONPATH:/path/to/han/src
```

Install dependencies:

```
$ pipenv sync
```

## Usage

### Step 1: Download data sets

Download the document classification data sets from my Google Drive [folder](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?usp=sharing). Unpack it somewhere to create the following directory structure:
```
/path/to/data
├── ag_news_csv
│   ├── classes.txt
│   ├── readme.txt
│   ├── test.csv
│   ├── train.csv
├── yahoo_answers_csv
│   ├── classes.txt
│   ├── readme.txt
│   ├── test.csv
│   ├── train.csv
...
```

### Step 2: Run the experiments

Every experiment has its own config file in `experiments`.
The pipeline of working with any model version or dataset is: 

``` bash
python run.py preprocess experiment_config_file   # Step 3a: preprocess the data
python run.py train experiment_config_file        # Step 3b: train a model
python run.py infer experiment_config_file        # Step 3c: evaluate the results
```

Use the following experiment config files to reproduce results:

* AG News, BiLSTM (GloVE embeddings) version: `experiments/han-yahoo-glove-run.jsonnet.jsonnet`
* AG News, BERT (base) version: `experiments/han-yahoo-bert-run.jsonnet.jsonnet`
* Yahoo Answers, BiLSTM (GloVE embeddings) version: `experiments/han-yahoo-glove-run.jsonnet`

One may add new configuration files from other data sets or even play with the hyper-parameters of the existing configuration.

The `infer` step will output the classification report against the test set of the desired data set.
For example, on the `AG News` data set, with BiLSTM (GloVE embeddings) sentence encoder:

```
               precision    recall  f1-score   support

       World       0.94      0.93      0.93      1900
      Sports       0.98      0.99      0.98      1900
    Business       0.89      0.91      0.90      1899
    Sci/Tech       0.92      0.90      0.91      1900

    accuracy                           0.93      7599
   macro avg       0.93      0.93      0.93      7599
weighted avg       0.93      0.93      0.93      7599
```

### Step 3: Visualize Predictions

One can visualize the sentence/word attention weights per each item in the test set, after running the `infer` command,
using the notebook `notebooks/Prediction Visualizer.ipynb`.

Please note that one may need to change the value of `PREDICTIONS_PATH` when using this notebook.

For example, for item in index 200, we will notice that the 2nd sentence (out of 2) got the most attention and same goes for
the phrases: **broadband  users** and **internet  users** that had the highest weights when determining the prediction of
class `Sci/Tech`:

![attention](./images/attention_200.jpg)


## References

[1] Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy, [Hierarchical Attention Networks for Document Classification
](https://www.aclweb.org/anthology/N16-1174/)

```
@inproceedings{yang-etal-2016-hierarchical,
    title = "Hierarchical Attention Networks for Document Classification",
    author = "Yang, Zichao  and
      Yang, Diyi  and
      Dyer, Chris  and
      He, Xiaodong  and
      Smola, Alex  and
      Hovy, Eduard",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N16-1174",
    doi = "10.18653/v1/N16-1174",
    pages = "1480--1489",
}
```

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, [{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding
](https://www.aclweb.org/anthology/N19-1423/)
```
@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1423",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
    abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7{\%} (4.6{\%} absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).",
}
```