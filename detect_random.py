#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""dockerfile_generator

Based on https://github.com/rrenaud/Gibberish-Detector

Detects if a filename is a randomly generated. Typically basename, extension, folders shall be tested 
separately
Train the system first by running the script with -d -m flags
 
Usage:
  detect_random.py -h | --help
  detect_random.py [-d <FILENAME>] [-o <PATH>] [-c <STR>]

Example:
    detect_random.py [-d training_data.txt] [-m model.pki]
   
Options:
  -h --help                 Show this screen.
  -d --data=<FILENAME>      File containing training data
  -m --model=<FILENAME>     File containing a model. Run the script with -d to generate the model
  -c --check=<STR>          Test a single string
"""

import logging
import sys
from docopt import docopt
import math
import pickle

global model_data
global model_mat
global threshold

def init(model_data_filename):
    global model_data
    global model_mat
    global threshold
    model_data = pickle.load(open(model_data_filename, 'rb'))
    model_data['mat']
    model_data['thresh']

accepted_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._-'

pos = dict([(char, idx) for idx, char in enumerate(accepted_chars)])

def normalize(line):
    """ Return only the subset of chars from accepted_chars.
    This helps keep the  model relatively small by ignoring punctuation, 
    infrequenty symbols, etc. """
    return [c.lower() for c in line if c.lower() in accepted_chars]

def ngram(n, l):
    """ Return all n grams from l after normalizing """
    filtered = normalize(l)
    for start in range(0, len(filtered) - n + 1):
        yield ''.join(filtered[start:start + n])

def avg_transition_prob(l, log_prob_mat):
    """ Return the average transition prob from l through log_prob_mat. """
    log_prob = 0.0
    transition_ct = 0
    for a, b in ngram(2, l):
        log_prob += log_prob_mat[pos[a]][pos[b]]
        transition_ct += 1
    # The exponentiation translates from log probs to probs.
    return math.exp(log_prob / (transition_ct or 1))

def train(logger, f):
    """ Write a simple model as a pickle file """
    k = len(accepted_chars)
    # Assume we have seen 2 of each character pair.  This acts as a kind of
    # prior or smoothing factor.  This way, if we see a character transition
    # live that we've never observed in the past, we won't assume the entire
    # string has 0 probability.
    counts = [[2 for i in xrange(k)] for i in xrange(k)]

    # Count transitions from big text file, taken 
    # from http://norvig.com/spell-correct.html
    for line in f:
        for a, b in ngram(2, line):
            counts[pos[a]][pos[b]] += 1

    # Normalize the counts so that they become log probabilities.  
    # We use log probabilities rather than straight probabilities to avoid
    # numeric underflow issues with long texts.
    # This contains a justification:
    # http://squarecog.wordpress.com/2009/01/10/dealing-with-underflow-in-joint-probability-calculations/
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j in xrange(len(row)):
            row[j] = math.log(row[j] / s)

    # Find the probability of generating a few arbitrarily choosen good and
    # bad phrases.
    good_probs = [avg_transition_prob(l, counts) for l in open('good.txt')]
    bad_probs = [avg_transition_prob(l, counts) for l in open('bad.txt')]

    # Assert that we actually are capable of detecting the junk.
    assert min(good_probs) > max(bad_probs)

    # And pick a threshold halfway between the worst good and best bad inputs.
    thresh = (min(good_probs) + max(bad_probs)) / 2
    return counts, thresh
    
def check(s):
    probability = avg_transition_prob(s, model_mat)
    return probability > threshold

def check_and_print(logger, s):
    probability = avg_transition_prob(s, model_mat)
    logger.info("{0}, probability {1}, threshold {2}",  probability > threshold, probability, threshold)
        
def open_file(filename, flags):
    '''
    Open a text file 
    Returns handle to the open file and result code False/True
    '''
    try:
        fileHandle = open(filename, flags) 
    except Exception:
        print sys.exc_info()
        logger.error('Failed to open file {0}'.format(filename))
        return (False, None)
    else:
        return (True, fileHandle)    


if __name__ == '__main__':
    '''
    When running as a script I assume training mode
    '''
    arguments = docopt(__doc__, version='0.1')
    logging.basicConfig()    
    logger = logging.getLogger('detect_random')
    logger.setLevel(logging.INFO)    
    
    data_filename = arguments['--data']
    output_filename = arguments['--output']
    check = arguments['--check']
        
        
    while True:
        if check is not None and output_filename is not None:
            init(output_filename)
            check_and_print(logger, check)
            break
        
        if data_filename is None or output_filename is None:
            logger.error("Both data and output filenames should be provided in the training mode")
        
        result, output_file = open_file(output_filename, "wb")
        if not result:
            logger.error("Failed to open {0} for writing".format(output_filename))
            break
        result, data_file = open_file(data_filename, "r")
        if not result:
            logger.error("Failed to open {0} for reading".format(data_filename))
            break
        counts, thresh = train(logger, data_file)
        pickle.dump({'mat': counts, 'thresh': thresh}, output_file)
        break;
    
    if output_file:  
        output_file.close()
    if data_file:
        data_file.close()
    
