#!/usr/bin/env python
"""
LDfeatureselect.py - 
LD (Lang-Domain) feature extractor
Marco Lui November 2011

Based on research by Marco Lui and Tim Baldwin.

Copyright 2011 Marco Lui <saffsd@gmail.com>. All rights reserved.

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
FEATURES_PER_LANG = 300 # number of features to select for each language

import os, sys, argparse
import csv
import marshal
import numpy
import multiprocessing as mp
from collections import defaultdict

from common import read_weights, Enumerator, write_features

def select_LD_features(ig_lang, ig_domain, feats_per_lang, ignore_domain=False):
  """
  @param ignore_domain boolean to indicate whether to use domain weights
  """
  assert (ig_domain is None) or (len(ig_lang) == len(ig_domain))
  num_lang = len(ig_lang.values()[0])
  num_term = len(ig_lang)

  term_index = defaultdict(Enumerator())


  ld = numpy.empty((num_lang, num_term), dtype=float)

  for term in ig_lang:
    term_id = term_index[term]
    if ignore_domain:
      ld[:, term_id] = ig_lang[term]
    else:
      ld[:, term_id] = ig_lang[term] - ig_domain[term]

  terms = sorted(term_index, key=term_index.get)
  # compile the final feature set
  selected_features = dict()
  for lang_id, lang_w in enumerate(ld):
    term_inds = numpy.argsort(lang_w)[-feats_per_lang:]
    selected_features[lang_id] = [terms[t] for t in term_inds]

  return selected_features
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-o","--output", metavar="OUTPUT_PATH", help = "write selected features to OUTPUT_PATH")
  parser.add_argument("--feats_per_lang", type=int, metavar='N', help="select top N features for each language", default=FEATURES_PER_LANG)
  parser.add_argument("--per_lang", action="store_true", default=False, help="produce a list of features selecter per-language")
  parser.add_argument("--no_domain_ig", action="store_true", default=False, help="use only per-langugage IG in LD calculation")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")
  args = parser.parse_args()

  lang_w_path = os.path.join(args.model, 'IGweights.lang.bin')
  domain_w_path = os.path.join(args.model, 'IGweights.domain')
  feature_path = args.output if args.output else os.path.join(args.model, 'LDfeats')

  # display paths
  print "model path:", args.model
  print "lang weights path:", lang_w_path
  print "domain weights path:", domain_w_path
  print "feature output path:", feature_path

  lang_w = read_weights(lang_w_path)
  domain_w = read_weights(domain_w_path) if not args.no_domain_ig else None

  features_per_lang = select_LD_features(lang_w, domain_w, args.feats_per_lang, ignore_domain=args.no_domain_ig)
  if args.per_lang:
    with open(feature_path + '.perlang', 'w') as f:
      writer = csv.writer(f)
      for i in range(len(features_per_lang)):
        writer.writerow(map(repr,features_per_lang[i]))
      

  final_feature_set = reduce(set.union, map(set, features_per_lang.values()))
  print 'selected %d features' % len(final_feature_set)

  write_features(sorted(final_feature_set), feature_path)
  print 'wrote features to "%s"' % feature_path 

