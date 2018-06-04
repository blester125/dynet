"""Conv-bLSTM-CRF tagging with Dynet."""

import os
import math
import random
import argparse
from itertools import chain
import dynet as dy
import numpy as np
from crf import CRF
from utils import read_conll, read_glove, Vocab, batch, f1, get_logger, create_conll_file

logger = get_logger('tagger')

# Get data location from command line args.
parser = argparse.ArgumentParser("A Tagger in Dynet")
parser.add_argument('--ner', '-n', default='ner')
parser.add_argument('--glove', '-g', default='glove')
args = parser.parse_known_args()[0]

# Process input
train = os.path.join(args.ner, 'eng.train')
dev = os.path.join(args.ner, 'eng.testa')
test = os.path.join(args.ner, 'eng.testb')
glove = os.path.join(args.glove, 'glove.6B.100d.txt')

# Read data
train_text, train_labels = read_conll(train)
logger.info("Size of training set: %s" % len(train_text))
dev_text, dev_labels = read_conll(dev)
logger.info("Size of dev set: %s" % len(dev_text))
test_text, test_labels = read_conll(test)
logger.info("Size of testing set: %s" % len(test_text))

# Create Vocab
vocab = Vocab.from_corpus(list(chain(train_text, dev_text, test_text)))
logger.info("Size of vocab: %s" % len(vocab))
l_vocab = Vocab.from_corpus(train_labels, unk=False, lower=False)
logger.info("Number of tags: %s" % len(l_vocab))
c_vocab = Vocab.from_corpus(list(chain(chain(*train_text), chain(*dev_text), chain(*test_text))), lower=False)
logger.info("Size of char vocab: %s" % len(c_vocab))

# HyperParameters
cvsz = len(c_vocab)
vsz = len(vocab)
n_tags = len(l_vocab)
unif = 0.1
cdsz = 30
cscale = math.sqrt(3 / float(cdsz))
dsz = 100
filtsz = [3]
cmotsz = 30
rnninsz = dsz + (cmotsz * len(filtsz))
rnnsz = 200
brnnsz = rnnsz * 2
rnn_layers = 1
dropout = 0.5
# Train Parameters
epochs = 100
eta = 0.015
mom = 0.9
decay = 0.05
clip = 5.0
batch_size = 10
patience = 40
train_every = 200
dev_every = 700

# Use pretrained vectors if available
vectors = None
if os.path.exists(glove):
    vectors = read_glove(glove, vocab, unif=unif)

# Model
pc = dy.ParameterCollection()

# Embeddings
embed = pc.add_subcollection(name="embeddings")
CEmbed = embed.add_lookup_parameters(
    (cvsz, 1, 1, cdsz),
    init=dy.UniformInitializer(cscale),
    name="char"
)

if vectors is None:
    WEmbed = embed.add_lookup_parameters((vsz, dsz), name="word")
else:
    WEmbed = embed.lookup_parameters_from_numpy(vectors, name="word")

# Char Conv
conv = pc.add_subcollection(name="conv")
convs = [conv.add_parameters((1, fsz, cdsz, cmotsz), name="w-%d" % fsz) for fsz in filtsz]
convs_bias = [conv.add_parameters(cmotsz, init=dy.ConstInitializer(0), name="b-%d" % fsz) for fsz in filtsz]

# LSTM
# blstm = dy.BiRNNBuilder(rnn_layers, rnninsz, brnnsz, pc, dy.LSTMBuilder)

class SumLSTM(object):
    def __init__(self, layers, isz, osz, pc):
        self._pc = pc.add_subcollection("sum-lstm")
        self.layers = layers
        self.isz = isz
        self.osz = osz
        self.W_x = self._pc.add_parameters((osz * 3, isz), name="weight-x")
        self.W_h = self._pc.add_parameters((osz * 2, osz), name="weight-h")
        self.b = self._pc.add_parameters((osz * 2), name="bias")

    def transduce(self, xs):
        osz = self.osz
        hs = []
        h_prev = dy.zeros((osz))
        c_prev = dy.zeros((osz))
        for x in xs:
            gate_h = self.W_h * h_prev
            gate_x = self.W_x * x
            c = dy.pickrange(gate_x, osz * 2, osz * 3)
            gate_x = dy.pickrange(gate_x, 0, osz * 2)
            gates = dy.logistic(gate_x + gate_h + self.b)
            i = dy.pickrange(gates, 0, osz)
            f = dy.pickrange(gates, osz, osz * 2)
            c_prev = dy.cmult(i, c) + dy.cmult(f, c_prev)
            h_prev = dy.tanh(c_prev)
            hs.append(h_prev)
        return hs

    def __str__(self):
        return "\n".join([f"{p.name()}: {p.shape()}" for p in self._pc.parameters_list()])

f_lstm = SumLSTM(rnn_layers, rnninsz, rnnsz, pc)
b_lstm = SumLSTM(rnn_layers, rnninsz, rnnsz, pc)

# Output Projection
proj = pc.add_subcollection(name="output")
W = proj.add_parameters((n_tags, brnnsz), name="weight")
b = proj.add_parameters(n_tags, init=dy.ConstInitializer(0), name="bias")

# CRF
crf = CRF(n_tags, l_vocab, pc)

def variational_dropout(input_, dropout):
    """Apply the same dropout mask to each step in a sequence."""
    pkeep = 1 - dropout
    mask = dy.random_bernoulli(input_[0].dim()[0], pkeep, (1 / pkeep))
    return [dy.cmult(inp, mask) for inp in input_]

def char_comp(word, train=True):
    """Create a representation base on character composition with a Conv net."""
    logger.debug("Word Length: %d" % len(word))
    logger.debug("Word: %s" % word)
    logger.debug("Encoded: %s" % [c_vocab[c] for c in word])
    c_embed = dy.concatenate([CEmbed[c_vocab[c]] for c in word], d=1)
    logger.debug("Char Embed Shape: %s" % (c_embed.dim(),))
    mots = []
    strides = (1, 1, 1, 1)
    for conv, bias in zip(convs, convs_bias):
        out = dy.tanh(dy.conv2d_bias(c_embed, conv, bias, strides, is_valid=False))
        logger.debug("Char Conv Shape: %s" % (out.dim(),))
        mot = dy.reshape(dy.max_dim(out, d=1), (cmotsz,))
        mots.append(mot)
    mot = dy.concatenate(mots)
    if train:
        mot = dy.dropout(mot, dropout)
    logger.debug("Max Overtime Shape: %s" % (mot.dim(),))
    return mot

def forward_emission(input_, train=True):
    """Calculate forward to produce emission probabilities."""
    logger.debug("Sequence Length: %d" % len(input_))
    logger.debug("Input: %s" % input_)
    logger.debug("Encoded: %s" % [vocab[x] for x in input_])
    char_embed = [char_comp(x, train) for x in input_]
    word_embed = [WEmbed[vocab[x]] for x in input_]
    inputs_ = [dy.concatenate([c, w]) for c, w in zip(char_embed, word_embed)]
    logger.debug("Input Shape: %s" % (inputs_[0].dim(),))
    # lstm_out = blstm.transduce(inputs_)
    f_out = f_lstm.transduce(inputs_)
    b_out = b_lstm.transduce(reversed(inputs_))
    lstm_out = [dy.concatenate([f, b]) for f, b in zip(f_out, reversed(b_out))]

    if train:
        lstm_out = variational_dropout(lstm_out, dropout)
    logger.debug("LSTM Shape: %s" % (lstm_out[0].dim(),))
    emissions = [dy.affine_transform([b, W, x]) for x in lstm_out]
    logger.debug("Emission Shape: %s" % (emissions[0].dim(),))
    return emissions

def calc_loss(input_, tags):
    """Calculate the CRF loss of a sample."""
    tags = [l_vocab[tag] for tag in tags]
    emissions = forward_emission(input_)
    return crf.neg_log_loss(emissions, tags)

def evaluate_loss(text, labels, set_):
    """Calculate the average loss on the dev/test set."""

    losses = []
    for t, l in zip(text, labels):
        l = [l_vocab[tag] for tag in l]
        emissions = forward_emission(t, train=False)
        losses.append(crf.neg_log_loss(emissions, l))
    res = dy.esum(losses).npvalue() / len(text)
    logger.info("%s Loss: %f" % (set_, res))

def predict(text):
    preds = []
    for t in text:
        dy.renew_cg()
        emissions = forward_emission(t, train=False)
        tags, s1 = crf.decode(emissions)
        preds.append([l_vocab.i2w[tag] for tag in tags])
    return preds

def evaluate(text, labels, set_):
    """Calculate F1 on the dev/test set."""
    preds = predict(text)
    f1_ = f1(labels, preds)
    logger.info("%s F1: %f" % (set_, f1_))
    return f1_


# Training
model_str = ["Tagger Model"]
for p in chain(pc.lookup_parameters_list(), pc.parameters_list()):
    model_str.append("\t %s: %s" % (p.name(), p.shape(),))
logger.info('\n'.join(model_str))

model_file = 'tagger.model'
result_file = 'results.conll'

# trainer = dy.MomentumSGDTrainer(pc, learning_rate=eta, mom=mom)
trainer = dy.SimpleSGDTrainer(pc, learning_rate=eta)
trainer.set_clip_threshold(clip)
# trainer.set_sparse_updates(False)

best_score = 0
last_best = 0

train = batch(train_text, train_labels, batch_size)

for epoch in range(epochs):
    logger.info("Epoch: %d" % epoch)
    random.shuffle(train)
    trainer.learning_rate = eta / (1 + decay * epoch)
    logger.info("Learning rate: %f" % trainer.learning_rate)
    total_loss = 0
    for i, batch in enumerate(train):
        dy.renew_cg()
        losses = []
        for t, l in batch:
            losses.append(calc_loss(t, l))
        loss = dy.esum(losses) / batch_size
        total_loss += loss.npvalue()
        loss.backward()
        trainer.update()
        # Print loss updates occasionally
        if (i + 1) % train_every == 0:
            logger.info("Train Loss: %f" % (total_loss / train_every))
            total_loss = 0
        if (i + 1) % dev_every == 0:
            evaluate_loss(dev_text, dev_labels, "Dev")

    score = evaluate(dev_text, dev_labels, "Dev")
    # Early stopping
    if score > best_score:
        logger.info("Best metric beaten at epoch: %d. Old %f, New %f" % (epoch, best_score, score))
        best_score = score
        pc.save(model_file)
        last_best = epoch
    else:
        logger.info("Best f1 was %f at epoch %d" % (best_score, last_best))
    # Patience
    if epoch - last_best > patience:
        logger.info("Stopping at epoch %d due to failure to improve." % epoch)
        break

pc.populate(model_file)
evaluate(test_text, test_labels, "Test")

# Create Conll results file.
logger.info("Creating conll results file at %s" % result_file)
create_conll_file(test_text, predict(test_text), test_labels, result_file)

try:
    from subprocess import run, PIPE
    res = run("perl conlleval.pl < {}".format(result_file), shell=True, stdout=PIPE)
    logger.info("conlleval.pl results:")
    print(res.stdout.decode('utf-8'))
except ImportError:
    logger.warn("Run `perl conlleval.pl < %s` for more detailed results." % result_file)
