#!/usr/bin/env bash
#
# Runs Stanford CoreNLP server

# set this path to the directory where you decompressed StanfordCore

# Dani posa el teu custom path :)
STANFORDDIR=~/Desktop/College/stanford-corenlp-4.5.9

if [ -f /tmp/corenlp-server.running ]; then
    echo "server already running"
else
    echo java -mx5g -cp \"$STANFORDDIR/*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $*
    java -mx5g -cp "$STANFORDDIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer $* &
    echo $! > /tmp/corenlp-server.running
    wait
    rm /tmp/corenlp-server.running
fi
