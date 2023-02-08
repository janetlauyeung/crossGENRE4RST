#!/usr/bin/env bash
#
# Runs Stanford CoreNLP.
# Simple uses for xml and plain text output to files are:
#    ./corenlp.sh -file filename
#    ./corenlp.sh -file filename -outputFormat text

scriptdir=`dirname $0`

# $1 - path

PATH=$1
for FNAME in $PATH/*
do
    if [[ "$FNAME" == *text ]]
    then
        /usr/bin/java -Xmx40g -cp "$scriptdir/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -file $FNAME -outputDirectory $PATH -outputExtension '.xml' -timeout 120000
    fi
done
