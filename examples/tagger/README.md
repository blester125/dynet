# Dynet Tagger

An example NER tagger built in Dynet reimplementing [Ma and Hovy, 2016](https://arxiv.org/pdf/1603.01354.pdf).

### Architecture

This tagger is a (hopefully) straight forward implementation of a Conv-bLSTM-CRF tagger.

There are only a few differences that I am aware of is this implementation:
 * It uses `IOB2` tags rather than `IOBES` but this shouldn't matter much according to [Lample et al., 2017](https://arxiv.org/pdf/1603.01360.pdf).
 * It uses Variational dropout for rnn layer outputs.
 * It does dropout after the Character composition rather than after the embeddings.
 * It uses a mask on CRF transitions.


### Performance

As discussed in [Reimers and Gurevych, 2017](https://arxiv.org/pdf/1707.09861.pdf) the random seed is very important for this model.

### Data

This example needs the Conll 2003 NER dataset and can use pretrained GloVe embeddings if available. These are assumed to be in directories called `ner` and `glove` respectively but can be changed with the `--ner` and `--glove` command line arguments.

If you don't have the data you can get it with the `.get_data.sh`. This accepts a command line argument that is the location of the Reuters datasets (`rcv1.tar.xz`). This data can be obtained from [here](https://trec.nist.gov/data/reuters/reuters.html). If you don't have the Reuters dataset it will try to get the conll data from elsewhere. This script will create the dataset and download glove vectors to use as well as downloading the `conlleval.pl` script.
