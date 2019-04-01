import os
import logging
from itertools import chain
from collections import defaultdict
import numpy as np


def get_logger(name, level=None):
    """Create a logger that can get level from env variable."""
    logger = logging.getLogger(name)
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(level)
    logger.propagate = False
    f = logging.Formatter('[%(name)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(f)
    logger.addHandler(sh)
    return logger


logger = get_logger('eval', 'WARN')


def read_glove(file_name, vocab, unif=0.25, dsz=100):
    """Read vectors from file and initialize unknown vectors to random uniform."""
    vectors = np.random.uniform(-unif, unif, size=(len(vocab), dsz))
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip("\n")
            parts = line.split()
            word = parts[0]
            parts = parts[1:]
            if len(parts) != dsz:
                continue
            if word in vocab:
                vectors[vocab[word], :] = np.array([float(x) for x in parts])
    return vectors


def convert_BIO(seq):
    """Convert an IOB sequence to IOB2 (BIO)."""
    new_seq = []
    prev = "O"
    for token in seq:
        if token.startswith('I-'):
            # Start entities with B-
            if prev == "O":
                new_seq.append("B" + token[1:])
            else:
                prev_type = prev.split("-")[1]
                curr_type = token.split("-")[1]
                # Start entities when changing types with B.
                if curr_type != prev_type:
                    new_seq.append("B" + token[1:])
                # Leave I- that continue an entity alone.
                else:
                    new_seq.append(token)
        # Leave B- and O alone
        else:
            new_seq.append(token)
        prev = token
    return new_seq


def read_conll(file_name, convert=convert_BIO):
    """Read dataset from a CONLL file and convert to BIO while skipping the DOCSTART lines."""
    sentences = []
    labels = []
    sentence = []
    label = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip("\n")
            if line.startswith("-DOCSTART-"):
                continue
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(convert(label))
                    sentence = []
                    label = []
                continue
            parts = line.split()
            sentence.append(parts[0])
            label.append(parts[-1])
    if sentence:
        sentences.append(sentence)
        labels.append(convert(label))
    return sentences, labels

def batch(text, labels, bsz=10):
    """Create batches that are grouped by length to speed things up."""
    zipped = list(zip(text, labels))
    zipped = sorted(zipped, key=lambda x: len(x[0]))
    batched = [zipped[x:x+bsz] for x in range(0, len(zipped), bsz)]
    return batched


class Vocab(object):
    """Vocab with options so that it is easy to use with words and characters."""
    UNK = '<UNK>'

    def __init__(self, w2i, lower=True):
        super(Vocab, self).__init__()
        self.w2i = dict(w2i)
        self.lower = lower
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus, unk=True, lower=True):
        """Build a vocab from a corpus.

        Corpus as a list of list of words will create a word level vocab.
        Corpus as a list of words with create a character level vocab.

        if `unk` then a special unk token is added to the vocab
        if `lower` then inputs are lowercased before being looked up
        """
        w2i = defaultdict(lambda: len(w2i))
        for word in chain(*corpus):
            word = word.lower() if lower else word
            w2i[word]
        if unk:
            w2i[Vocab.UNK]
        return Vocab(w2i, lower)

    def __getitem__(self, item):
        item = item.lower() if self.lower else item
        if item in self.w2i:
            return self.w2i[item]
        return self.w2i[Vocab.UNK]

    def __iter__(self):
        return self.w2i.__iter__()

    def __len__(self):
        return len(self.w2i)

    def __contains__(self, item):
        return item in self.w2i


def f1(gold, preds):
    """Calculate entity level F1 on a list of sequences."""
    correct_count = 0
    pred_count = 0
    gold_count = 0
    for g, p in zip(gold, preds):
        g = set(to_entities(g))
        p = set(to_entities(p))
        correct_count += len(g & p)
        pred_count += len(p)
        gold_count += len(g)
    if pred_count == 0:
        return 0.0
    p = correct_count / float(pred_count)
    r = correct_count / float(gold_count)
    if p == 0 or r == 0:
        return 0.0
    f = (2 * p * r) / (p + r)
    return f


def to_entities(labels):
    """Convert a sequence of labels in BIO to entities."""
    entities = []
    type_ = None
    start = None

    for i, token in enumerate(labels):
        if token.startswith('I-'):
            # If you are part of an entity
            if type_:
                # If the types match
                token_type = token.split('-')[1]
                if type_ != token_type:
                    logger.warning("Illegal transition from %s to %s" % (type_, token_type))
                    # Even though it is illegal for BIO, transition to match conlleval
                    entities.append((type_, start, i))
                    type_ = token_type
                    start = i
            else:
                logger.warning("Illegal start of a entity %s" % token)
                type_ = token.split('-')[1]
                start = i
        # Start of entity
        elif token.startswith('B-'):
            # If we are ending the prev entity
            if type_:
                entities.append((type_, start, i))
            # Start a new entity
            type_ = token.split('-')[1]
            start = i
        # End of an entity due to O
        else:
            if type_:
                entities.append((type_, start, i))
            type_ = None
    # If you end a seq in an entity
    if type_ is not None:
        entities.append((type_, start, i + 1))
    return entities


def create_conll_file(text, preds, golds, file_name="results.conll"):
    with open(file_name, 'w') as f:
        for sentence, pred, gold in zip(text, preds, golds):
            example = ["{} {} {}".format(s, p, g) for s, p, g in zip(sentence, pred, gold)]
            section = "\n".join(example)
            f.write(section + "\n\n")
