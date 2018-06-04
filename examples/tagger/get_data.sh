# CONLL
NERURL="https://www.clips.uantwerpen.be/conll2003/ner.tgz"
NERTAR="ner.tgz"
NERDIR="ner"

TRAIN="ner/eng.train"
DEV="ner/eng.testa"
TEST="ner/eng.testb"
RCV1=${1:-"rcv1.tar.xz"}

GITURL="https://github.com/Franck-Dernoncourt/NeuroNER/archive/master.zip"
GITNAME="NeuroNER-master"


# If you don't have the data
if [ ! -e $TRAIN ] || [ ! -e $DEV ] || [ ! -e $TEST ]; then
    echo "Getting NER data..."
    # You have Reuters data.
    if [ -e $RCV1 ]; then
        if [ ! -e $NERDIR ]; then
            if [ ! -e $NERTAR ]; then
                wget --quiet $NERURL
            fi
            tar -xzf $NERTAR
        fi

        cp $RCV1 $NERDIR/$RCV1

        cd $NERDIR
        ./bin/make.eng.2016
        cd ..
    else
        mkdir -p $NERDIR
        wget --quiet $GITURL -O "master.zip"
        unzip -o -qq "master.zip"
        mv "$GITNAME/data/conll2003/en/test.txt" $TEST
        mv "$GITNAME/data/conll2003/en/valid.txt" $DEV
        mv "$GITNAME/data/conll2003/en/train.txt" $TRAIN
        rm -rf $GITNAME
        rm -rf "master.zip"
    fi
fi

GLOVEDIR="glove"
GLOVEURL="http://nlp.stanford.edu/data/glove.6B.zip"
GLOVEZIP="glove.6B.zip"
GLOVEFILE="glove/glove.6B.100d.txt"

if [ ! -e $GLOVEFILE ]; then
    echo "Getting Glove data..."
    if [ ! -e $GLOVEZIP ]; then
        wget --quiet $GLOVEURL
    fi
    unzip -qq $GLOVEZIP -d $GLOVEDIR
fi

CONLLEVALURL="http://deeplearning.net/tutorial/code/conlleval.pl"
CONLLEVAL="conlleval.pl"

if [ ! -e $CONLLEVAL ]; then
    echo "Getting conlleval script..."
    wget --quiet $CONLLEVALURL
fi
