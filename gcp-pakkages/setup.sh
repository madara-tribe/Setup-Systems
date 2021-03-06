# !/bin/sh
ENV=sample 
find . -name '.DS_Store' -type f -ls -delete
#!pip3 install virtualenv
virtualenv ${ENV}
