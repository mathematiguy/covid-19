#! /bin/bash

set -ex

make report

cp data/* /output
cp stan/*.rds /output
cp -r _book/* /publish
