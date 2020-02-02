#!/bin/bash

TMP_DIR=`mktemp -d`
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip -P $TMP_DIR/
unzip -d resources/ $TMP_DIR/Banana_Linux.zip
rm -r $TMP_DIR

