#!/usr/bin/env python
"""
IGWeight.py - 
Compute IG Weights given a set of tokenized buckets and a feature set

Marco Lui, January 2013

Based on research by Marco Lui and Tim Baldwin.

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

import os, sys, argparse 
import csv
import numpy
import multiprocessing as mp
from itertools import tee, imap, islice
from collections import defaultdict
from contextlib import closing

from common import unmarshal_iter, MapPool, Enumerator, write_weights, read_features 

def entropy(v, axis=0):
  """
  Optimized implementation of entropy. This version is faster than that in 
  scipy.stats.distributions, particularly over long vectors.
  """
  v = numpy.array(v, dtype='float')
  s = numpy.sum(v, axis=axis)
  with numpy.errstate(divide='ignore', invalid='ignore'):
    rhs = numpy.nansum(v * numpy.log(v), axis=axis) / s
    r = numpy.log(s) - rhs
  # Where dealing with binarized events, it is possible that an event always
  # occurs and thus has 0 information. In this case, the negative class
  # will have frequency 0, resulting in log(0) being computed as nan.
  # We replace these nans with 0
  nan_index = numpy.isnan(rhs)
  if nan_index.any():
    r[nan_index] = 0
  return r

def setup_pass_IG(features, dist, binarize, suffix):
  """
  @param features the list of features to compute IG for
  @param dist the background distribution
  @param binarize (boolean) compute IG binarized per-class if True
  @param suffix of files in bucketdir to process
  """
  global __features, __dist, __binarize, __suffix
  __features = features
  __dist = dist
  __binarize = binarize
  __suffix = suffix

def pass_IG(buckets):
  """
  In this pass we compute the information gain for each feature, binarized 
  with respect to each language as well as unified over the set of all 
  classes. 

  @global __features the list of features to compute IG for
  @global __dist the background distribution
  @global __binarize (boolean) compute IG binarized per-class if True
  @global __suffix of files in bucketdir to process
  @param buckets a list of buckets. Each bucket must be a directory that contains files 
                 with the appropriate suffix. Each file must contain marshalled 
                 (term, event_id, count) triplets.
  """
  global __features, __dist, __binarize, __suffix
   
  # We first tally the per-event frequency of each
  # term in our selected feature set.
  term_freq = defaultdict(lambda: defaultdict(int))
  term_index = defaultdict(Enumerator())

  for bucket in buckets:
		for path in os.listdir(bucket):
			if path.endswith(__suffix):
				for key, event_id, count in unmarshal_iter(os.path.join(bucket,path)):
					# Select only our listed features
					if key in __features:
						term_index[key]
						term_freq[key][event_id] += count

  num_term = len(term_index)
  num_event = len(__dist)

  cm_pos = numpy.zeros((num_term, num_event), dtype='int')

  for term,term_id in term_index.iteritems():
    # update event matrix
    freq = term_freq[term]
    for event_id, count in freq.iteritems():
      cm_pos[term_id, event_id] = count
  cm_neg = __dist - cm_pos
  cm = numpy.dstack((cm_neg, cm_pos))

  if not __binarize:
    # non-binarized event space
    x = cm.sum(axis=1)
    term_w = x / x.sum(axis=1)[:, None].astype(float)

    # Entropy of the term-present/term-absent events
    e = entropy(cm, axis=1)

    # Information Gain with respect to the set of events
    ig = entropy(__dist) - (term_w * e).sum(axis=1)

  else:
    # binarized event space
    # Compute IG binarized with respect to each event
    ig = list()
    for event_id in xrange(num_event):
      num_doc = __dist.sum()
      prior = numpy.array((num_doc - __dist[event_id], __dist[event_id]), dtype=float) / num_doc

      cm_bin = numpy.zeros((num_term, 2, 2), dtype=int) # (term, p(term), p(lang|term))
      cm_bin[:,0,:] = cm.sum(axis=1) - cm[:,event_id,:]
      cm_bin[:,1,:] = cm[:,event_id,:]

      e = entropy(cm_bin, axis=1)
      x = cm_bin.sum(axis=1)
      term_w = x / x.sum(axis=1)[:, None].astype(float)

      ig.append( entropy(prior) - (term_w * e).sum(axis=1) )
    ig = numpy.vstack(ig)

  terms = sorted(term_index, key=term_index.get)
  return terms, ig


def compute_IG(bucketlist, features, dist, binarize, suffix, job_count=None):
  pass_IG_args = (features, dist, binarize, suffix)

  num_chunk = len(bucketlist)
  weights = []
  terms = []

  with MapPool(job_count, setup_pass_IG, pass_IG_args) as f:
    pass_IG_out = f(pass_IG, bucketlist)

    for i, (t, w) in enumerate(pass_IG_out):
      weights.append(w)
      terms.extend(t)
      print "processed chunk (%d/%d) [%d terms]" % (i+1, num_chunk, len(t))

  if binarize:
    weights = numpy.hstack(weights).transpose()
  else:
    weights = numpy.concatenate(weights)
  terms = ["".join(t) for t in terms]

  return zip(terms, weights)

def read_dist(path):
  """
  Read the distribution from a file containing item, count pairs.
  @param path path to read form
  """
  with open(path) as f:
    reader = csv.reader(f)
    return numpy.array(zip(*reader)[1], dtype=int)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-f","--features", metavar='FEATURE_FILE', help="read features from FEATURE_FILE")
  parser.add_argument("-w","--weights", metavar='WEIGHTS', help="output weights to WEIGHTS")
  parser.add_argument("-d","--domain", action="store_true", default=False, help="compute IG with respect to domain")
  parser.add_argument("-b","--binarize", action="store_true", default=False, help="binarize the event space in the IG computation")
  parser.add_argument("-l","--lang", action="store_true", default=False, help="compute IG with respect to language")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")
  parser.add_argument("buckets", nargs='*', help="read bucketlist from")

  args = parser.parse_args()
  if not(args.domain or args.lang) or (args.domain and args.lang):
    parser.error("exactly one of domain(-d) or language (-l) must be specified")

  if args.features:
    feature_path = args.features
  else:
    feature_path = os.path.join(args.model, 'DFfeats')

  if args.buckets:
    bucketlist_paths = args.buckets
  else:
    bucketlist_paths = [os.path.join(args.model, 'bucketlist')]

  if not os.path.exists(feature_path):
    parser.error('{0} does not exist'.format(feature_path))

  features = read_features(feature_path)

  if args.domain:
    index_path = os.path.join(args.model,'domain_index')
    suffix = '.domain'
  elif args.lang:
    index_path = os.path.join(args.model,'lang_index')
    suffix = '.lang'
  else:
    raise ValueError("no event specified")

  if args.weights:
    weights_path = args.weights
  else:
    weights_path = os.path.join(args.model, 'IGweights' + suffix + ('.bin' if args.binarize else ''))

  # display paths
  print "model path:", args.model 
  print "buckets path:", bucketlist_paths
  print "features path:", feature_path
  print "weights path:", weights_path
  print "index path:", index_path
  print "suffix:", suffix

  print "computing information gain"
  # Compile buckets together
  bucketlist = zip(*(map(str.strip, open(p)) for p in bucketlist_paths))

  # Check that each bucketlist has the same number of buckets
  assert len(set(map(len,bucketlist))) == 1, "incompatible bucketlists!"

  dist = read_dist(index_path)
  ig = compute_IG(bucketlist, features, dist, args.binarize, suffix, args.jobs)

  write_weights(ig, weights_path)
