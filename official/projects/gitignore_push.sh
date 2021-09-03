#!/bin/bash
M=$1

find . -size +50M | cat >> .gitignore

git add .
git commit -m $M 
git push 
