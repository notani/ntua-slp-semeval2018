#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Make predictions.

Example:

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python predict/task3/make_predictions.py trained/TASK3_A_word_0.7900.model -i datasets/task3/gold/SemEval2018-T3_gold_test_taskA_emoji.txt -v
"""

from os import path
import argparse
import logging

from dataloaders.task3 import parse_file
from predict.predictions import dump_attentions
from utils.train import load_pretrained_model

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def main(args):
    global verbose
    verbose = args.verbose

    if verbose:
        logger.info('Load ' + args.path_model)
    assert args.path_model.endswith('.model')
    path_model = path.basename(args.path_model)
    model, conf = load_pretrained_model(path_model.replace('.model', ''))

    if verbose:
        logger.info('Read ' + args.path_input)
    X, y = parse_file(args.path_input)
    path_output = path.basename(args.path_input).replace('.tsv', '')
    dump_attentions(X, y, path_output, model, conf, "bclf")

    return 0


if __name__ == '__main__':
    logger = init_logger('Pred')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_model', help='path to a model file [.model]')
    parser.add_argument('-i', '--input', dest='path_input',
                        help='path to an input file [.tsv]')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
