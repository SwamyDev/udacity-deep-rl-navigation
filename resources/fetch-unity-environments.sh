#!/bin/bash

TMP_DIR=`mktemp -d`
mkdir -p resources/environments
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip -P $TMP_DIR/
unzip -d resources/environments $TMP_DIR/Banana_Linux.zip
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip -P $TMP_DIR/
unzip -d resources/environments $TMP_DIR/Reacher_Linux.zip
rm -r $TMP_DIR

