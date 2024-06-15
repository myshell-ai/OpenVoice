#!/usr/bin/env python
"""
DFfeatureselect.py - 
First step in the LD feature selection process, select features based on document
frequency.

Marco Lui January 2013

Copyright 2013 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

######
# Default values
# Can be overriden with command-line options
######
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOKENS_PER_ORDER = 15000 # number of tokens to consider for each order

import os, sys, argparse
import collections
import csv
import shutil
import tempfile
import marshal
import random
import numpy
import cPickle
import multiprocessing as mp
import atexit
import gzip
from itertools import tee, imap, islice
from collections import defaultdict
from datetime import datetime
from contextlib import closing

from common import Enumerator, unmarshal_iter, MapPool, write_features, write_weights

def pass_sum_df(bucket):
  """
  Compute document frequency (df) by summing up (key,domain,count) triplets
  over all domains.
  """
  doc_count = defaultdict(int)
  count = 0
  with gzip.open(os.path.join(bucket, "docfreq"),'wb') as docfreq:
    for path in os.listdir(bucket):
      # We use the domain buckets as there are usually less domains
      if path.endswith('.domain'):
        for key, _, value in unmarshal_iter(os.path.join(bucket,path)):
          doc_count[key] += value
          count += 1
    
    for item in doc_count.iteritems():
      docfreq.write(marshal.dumps(item))
  return count

def tally(bucketlist, jobs=None):
  """
  Sum up the counts for each feature across all buckets. This
  builds a full mapping of feature->count. This is stored in-memory
  and thus could be an issue for large feature sets.
  """

  with MapPool(jobs) as f:
    pass_sum_df_out = f(pass_sum_df, bucketlist)

    for i, keycount in enumerate(pass_sum_df_out):
      print "processed bucket (%d/%d) [%d keys]" % (i+1, len(bucketlist), keycount)

  # build the global term->df mapping
  doc_count = {}
  for bucket in bucketlist:
    for key, value in unmarshal_iter(os.path.join(bucket, 'docfreq')):
      doc_count[key] = value

  return doc_count



def ngram_select(doc_count, max_order=MAX_NGRAM_ORDER, tokens_per_order=TOKENS_PER_ORDER):
  """
  DF feature selection for byte-ngram tokenization
  """
  # Work out the set of features to compute IG
  features = set()
  for i in range(1, max_order+1):
    d = dict( (k, doc_count[k]) for k in doc_count if len(k) == i)
    features |= set(sorted(d, key=d.get, reverse=True)[:tokens_per_order])
  features = sorted(features)
  
  return features



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-f","--features", metavar='FEATURE_FILE', help="output features to FEATURE_FILE")
  parser.add_argument("--tokens_per_order", metavar='N', type=int, help="consider top N tokens per ngram order")
  parser.add_argument("--tokens", metavar='N', type=int, help="consider top N tokens")
  parser.add_argument("--max_order", type=int, help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_argument("--doc_count", nargs='?', const=True, metavar='DOC_COUNT_PATH', help="output full mapping of feature->frequency to DOC_COUNT_PATH")
  parser.add_argument("--bucketlist", help="read list of buckets from")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")
  
  args = parser.parse_args()

  if args.tokens and args.tokens_per_order:
    parser.error("--tokens and --tokens_per_order are mutually exclusive")

  # if neither --tokens nor --tokens_per_order is given, default behaviour is tokens_per_order
  if not(args.tokens) and not(args.tokens_per_order):
    args.tokens_per_order = TOKENS_PER_ORDER
  
  if args.features:
    feature_path = args.features
  else:
    feature_path = os.path.join(args.model, 'DFfeats')

  if args.bucketlist:
    bucketlist_path = args.bucketlist 
  else:
    bucketlist_path = os.path.join(args.model, 'bucketlist')

  # display paths
  print "buckets path:", bucketlist_path
  print "features output path:", feature_path
  if args.tokens_per_order:
    print "max ngram order:", args.max_order
    print "tokens per order:", args.tokens_per_order
  else:
    print "tokens:", args.tokens

  with open(bucketlist_path) as f:
    bucketlist = map(str.strip, f)

  doc_count = tally(bucketlist, args.jobs)
  print "unique features:", len(doc_count)
  if args.doc_count:
    # The constant true is used to indicate output to default location
    doc_count_path = os.path.join(args.model, 'DF_all') if args.doc_count == True else args.doc_count
    write_weights(doc_count, doc_count_path)
    print "wrote DF counts for all features to:", doc_count_path

  if args.tokens_per_order:
    # Choose a number of features for each length of token
    feats = ngram_select(doc_count, args.max_order, args.tokens_per_order)
  else:
    # Choose a number of features overall
    feats = sorted( sorted(doc_count, key=doc_count.get, reverse=True)[:args.tokens] )
  print "selected features: ", len(feats)

  write_features(feats, feature_path)
  print 'wrote features to "%s"' % feature_path 

  
