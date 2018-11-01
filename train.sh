#!/usr/bin/env bash

src/trainer.py --rand-seed=100 --stage=train_analysts --num-epochs=50 | tee logs/train_analysts.txt &&
src/trainer.py --rand-seed=100 --stage=pretrain_manipulators --num-epochs=10 | tee logs/pretrain_manipulators.txt &&
src/trainer.py --rand-seed=100 --stage=train_manipulators --num-epochs=10 | tee logs/train_manipulators.txt
