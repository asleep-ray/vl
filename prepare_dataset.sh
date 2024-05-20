#!/bin/bash

# Loop from 0 to 63
for i in {0..31}
do
  echo python prepare_dataset.py -n $i &
  python prepare_dataset.py -n $i &
done
