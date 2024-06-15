#!/usr/bin/env python
"""
index.py - 
Index a corpus that is stored in a directory hierarchy as follows:

- corpus
  - domain1
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - domain2
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - ...

This produces 3 files: 
* index: a list of paths, together with the langid and domainid as integers
* lang_index: a list of languages in ascending order of id, with the count for each
* domain_index: a list of domains in ascending order of id, with the count for each

Marco Lui, January 2013

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
TRAIN_PROP = 1.0 # probability than any given document is selected
MIN_DOMAIN = 1 # minimum number of domains a language must be present in to be included

import os, sys, argparse
import csv
import random
import numpy
from itertools import tee, imap, islice
from collections import defaultdict

from common import Enumerator, makedir

class CorpusIndexer(object):
  """
  Class to index the contents of a corpus
  """
  def __init__(self, root, min_domain=MIN_DOMAIN, proportion=TRAIN_PROP, langs=None, domains=None, line_level=False):
    self.root = root
    self.min_domain = min_domain
    self.proportion = proportion 

    if langs is None:
      self.lang_index = defaultdict(Enumerator())
    else:
      # pre-specified lang set
      self.lang_index = dict((k,v) for v,k in enumerate(langs))

    if domains is None:
      self.domain_index = defaultdict(Enumerator())
    else:
      # pre-specified domain set
      self.domain_index = dict((k,v) for v,k in enumerate(domains))

    self.coverage_index = defaultdict(set)
    self.items = list()

    if os.path.isdir(root):
      # root supplied was the root of a directory structure
      candidates = []
      for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for docname in filenames:
          candidates.append(os.path.join(dirpath, docname))
    else:
      # root supplied was a file, interpet as list of paths
      candidates = map(str.strip, open(root))

    if line_level:
      self.index_line(candidates)
    else:
      self.index(candidates)

    self.prune_min_domain(self.min_domain)

  def index_line(self, candidates):
    """
    Line-level indexing. Assumes the list of candidates is file-per-class,
    where each line is a document.
    """
    if self.proportion < 1.0:
      raise NotImplementedError("proportion selection not available for file-per-class")

    for path in candidates:
      d, lang = os.path.split(path)
      d, domain = os.path.split(d)

      # index the language and the domain
      try:
        # TODO: If lang is pre-specified but not domain, we can end up 
        #       enumerating empty domains.
        domain_id = self.domain_index[domain]
        lang_id = self.lang_index[lang]
      except KeyError:
        # lang or domain outside a pre-specified set so
        # skip this document.
        continue

      # add the domain-lang relation to the coverage index
      self.coverage_index[domain].add(lang)

      with open(path) as f:
        for i,row in enumerate(f):
          docname = "line{0}".format(i)
          self.items.append((domain_id,lang_id,docname,path))

  def index(self, candidates):

    # build a list of paths
    for path in candidates:
      # Each file has 'proportion' chance of being selected.
      if random.random() < self.proportion:

        # split the path into identifying components
        d, docname = os.path.split(path)
        d, lang = os.path.split(d)
        d, domain = os.path.split(d)

        # index the language and the domain
        try:
          # TODO: If lang is pre-specified but not domain, we can end up 
          #       enumerating empty domains.
          domain_id = self.domain_index[domain]
          lang_id = self.lang_index[lang]
        except KeyError:
          # lang or domain outside a pre-specified set so
          # skip this document.
          continue

        # add the domain-lang relation to the coverage index
        self.coverage_index[domain].add(lang)

        # add the item to our list
        self.items.append((domain_id,lang_id,docname,path))

  def prune_min_domain(self, min_domain):
    # prune files for all languages that do not occur in at least min_domain 
     
    # Work out which languages to reject as they are not present in at least 
    # the required number of domains
    lang_domain_count = defaultdict(int)
    for langs in self.coverage_index.values():
      for lang in langs:
        lang_domain_count[lang] += 1
    reject_langs = set( l for l in lang_domain_count if lang_domain_count[l] < min_domain)

    # Remove the languages from the indexer
    if reject_langs:
      #print "reject (<{0} domains): {1}".format(min_domain, sorted(reject_langs))
      reject_ids = set(self.lang_index[l] for l in reject_langs)
    
      new_lang_index = defaultdict(Enumerator())
      lm = dict()
      for k,v in self.lang_index.items():
        if v not in reject_ids:
          new_id = new_lang_index[k]
          lm[v] = new_id

      # Eliminate all entries for the languages
      self.items = [ (d, lm[l], n, p) for (d, l, n, p) in self.items if l in lm]

      self.lang_index = new_lang_index


  @property
  def dist_lang(self):
    """
    @returns A vector over frequency counts for each language
    """
    retval = numpy.zeros((len(self.lang_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[l] += 1
    return retval

  @property
  def dist_domain(self):
    """
    @returns A vector over frequency counts for each domain 
    """
    retval = numpy.zeros((len(self.domain_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[d] += 1
    return retval

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--line", action="store_true",
      help="treat each line in a file as a document")
  parser.add_argument("-p","--proportion", type=float, default=TRAIN_PROP,
      help="proportion of training data to use" )
  parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
  parser.add_argument("-d","--domain", metavar="DOMAIN", action='append',
      help="use DOMAIN - can be specified multiple times (uses all domains found if not specified)")
  parser.add_argument("-l","--lang", metavar="LANG", action='append',
      help="use LANG - can be specified multiple times (uses all langs found if not specified)")
  parser.add_argument("--min_domain", type=int, default=MIN_DOMAIN,
      help="minimum number of domains a language must be present in" )
  parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")

  args = parser.parse_args()

  corpus_name = os.path.basename(args.corpus)
  if args.model:
    model_dir = args.model
  else:
    model_dir = os.path.join('.', corpus_name+'.model')

  makedir(model_dir)

  langs_path = os.path.join(model_dir, 'lang_index')
  domains_path = os.path.join(model_dir, 'domain_index')
  index_path = os.path.join(model_dir, 'paths')

  # display paths
  print "corpus path:", args.corpus
  print "model path:", model_dir
  print "writing langs to:", langs_path
  print "writing domains to:", domains_path
  print "writing index to:", index_path

  if args.line:
    print "indexing documents at the line level"

  indexer = CorpusIndexer(args.corpus, min_domain=args.min_domain, proportion=args.proportion,
                          langs = args.lang, domains = args.domain, line_level=args.line)

  # Compute mappings between files, languages and domains
  lang_dist = indexer.dist_lang
  lang_index = indexer.lang_index
  lang_info = ' '.join(("{0}({1})".format(k, lang_dist[v]) for k,v in lang_index.items()))
  print "langs({0}): {1}".format(len(lang_dist), lang_info)

  domain_dist = indexer.dist_domain
  domain_index = indexer.domain_index
  domain_info = ' '.join(("{0}({1})".format(k, domain_dist[v]) for k,v in domain_index.items()))
  print "domains({0}): {1}".format(len(domain_dist), domain_info)

  print "identified {0} documents".format(len(indexer.items))

  # output the language index
  with open(langs_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows((l, lang_dist[lang_index[l]]) 
        for l in sorted(lang_index.keys(), key=lang_index.get))

  # output the domain index
  with open(domains_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows((d, domain_dist[domain_index[d]]) 
        for d in sorted(domain_index.keys(), key=domain_index.get))

  # output items found
  with open(index_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows( sorted(set((d,l,p) for (d,l,n,p) in indexer.items)) )
