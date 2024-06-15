#!/usr/bin/env python
"""
tokenize.py - 
Tokenizer for langid.py training system. This takes a list of files and tokenizes them
in parallel.

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
MIN_NGRAM_ORDER = 1 # smallest order of n-grams to consider
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOP_DOC_FREQ = 15000 # number of tokens to consider for each order
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os, sys, argparse
import csv
import shutil
import marshal
import multiprocessing as mp
import random
import atexit
import gzip
import tempfile

from itertools import tee 
from collections import defaultdict, Counter

from common import makedir, chunk, MapPool

class NGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    min_order = self.min_order
    max_order = self.max_order
    t = tee(seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = ''.join(tn.next() for tn in t)
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield token[:n+1]
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield token[a:a+b]

class WordNGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    _seq = str.split(seq)
    min_order = self.min_order
    max_order = self.max_order
    t = tee(_seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = [tn.next() for tn in t]
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield ' '.join(token[:n+1])
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield ' '.join(token[a:a+b])

@atexit.register
def cleanup():
  global b_dirs, complete
  try:
    if not complete:
      for d in b_dirs:
        shutil.rmtree(d)
  except NameError:
    # Failed before globals defined, nothing to clean
    pass

def setup_pass_tokenize(tokenizer, b_dirs, sample_count, sample_size, term_freq, line_level):
  global __tokenizer, __b_dirs, __sample_count, __sample_size, __term_freq, __line_level
  __tokenizer = tokenizer
  __b_dirs = b_dirs
  __sample_count = sample_count
  __sample_size = sample_size
  __term_freq = term_freq
  __line_level = line_level

def pass_tokenize(chunk_items):
  """
  Computes the conditional frequencies of terms. The frequency can be
  either term frequency or document frequency, controlled by a global
  variable. Files are converted into a sequence of terms, which
  are then reduced to either TF or DF. The counts are redistributed to
  buckets via Python's built-in hash function. This is basically an
  inversion setp, so that the counts are now grouped by term rather
  than by document.
  """
  global __maxorder, __b_dirs, __tokenizer, __sample_count, __sample_size, __term_freq, __line_level
  
  extractor = __tokenizer
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))

  for domain_id, lang_id, path in chunk_items:
    with open(path) as f:
      if __sample_count:
        # sampling tokenization
        text = f.read()
        poss = max(1,len(text) - __sample_size) # possibe start locations
        count = min(poss, __sample_count) # reduce number of samples if document is too short
        offsets = random.sample(xrange(poss), count)
        for offset in offsets:
          tokens = extractor(text[offset: offset+__sample_size])
          if args.__term_freq:
            # Term Frequency
            tokenset = Counter(tokens)
          else:
            # Document Frequency
            tokenset = Counter(set(tokens))
          for token, count in tokenset.iteritems():
            term_lng_freq[token][lang_id] += count
            term_dom_freq[token][domain_id] += count
      elif __line_level:
        # line-model - each line in a file should be interpreted as a document
        for line in f:
          tokens = extractor(line)
          if __term_freq:
            # Term Frequency
            tokenset = Counter(tokens)
          else:
            # Document Frequency
            tokenset = Counter(set(tokens))
          for token, count in tokenset.iteritems():
            term_lng_freq[token][lang_id] += count
            term_dom_freq[token][domain_id] += count
          
      else:
        # whole-document tokenization
        tokens = extractor(f.read())
        if __term_freq:
          # Term Frequency
          tokenset = Counter(tokens)
        else:
          # Document Frequency
          tokenset = Counter(set(tokens))
        for token, count in tokenset.iteritems():
          term_lng_freq[token][lang_id] += count
          term_dom_freq[token][domain_id] += count

  # Output the counts to the relevant bucket files. 
  __procname = mp.current_process().name
  b_freq_lang = [gzip.open(os.path.join(p,__procname+'.lang'),'a') for p in __b_dirs]
  b_freq_domain = [gzip.open(os.path.join(p,__procname+'.domain'),'a') for p in __b_dirs]

  for term in term_lng_freq:
    bucket_index = hash(term) % len(b_freq_lang)
    for lang, count in term_lng_freq[term].iteritems():
      b_freq_lang[bucket_index].write(marshal.dumps((term, lang, count)))
    for domain, count in term_dom_freq[term].iteritems():
      b_freq_domain[bucket_index].write(marshal.dumps((term, domain, count)))

  # Close all the open files
  for f in b_freq_lang + b_freq_domain:
    f.close()

  return len(term_lng_freq)

def build_index(items, tokenizer, outdir, buckets=NUM_BUCKETS, 
        jobs=None, chunksize=CHUNKSIZE, sample_count=None, 
        sample_size=None, term_freq=False, line_level=False):
  """
  @param items a list of (domain, language, path) tuples
  """
  global b_dirs, complete

  # Our exitfunc uses this to know whether to delete the tokenized files
  complete = False 

  if jobs is None:
    jobs = mp.cpu_count() + 4

  b_dirs = [ os.path.join(outdir,"bucket{0}".format(i)) for i in range(buckets) ]

  for d in b_dirs:
    os.mkdir(d)

  # PASS 1: Tokenize documents into sets of terms
   
  # If there are few items, make the chunk size such that each job
  # will have 2 chunks
  chunk_size = max(1,min(len(items) / (jobs * 2), chunksize))
  item_chunks = list(chunk(items, chunk_size))
  pass_tokenize_globals = (tokenizer, b_dirs, sample_count, sample_size, term_freq, line_level)

  with MapPool(jobs, setup_pass_tokenize, pass_tokenize_globals) as f:
    pass_tokenize_out = f(pass_tokenize, item_chunks)


    doc_count = defaultdict(int)
    chunk_count = len(item_chunks)
    print "chunk size: {0} ({1} chunks)".format(chunk_size, chunk_count)
    print "job count: {0}".format(jobs)

    if sample_count:
      print "sampling-based tokenization: size {0} count {1}".format(sample_size, sample_count)
    else:
      print "whole-document tokenization"

    for i, keycount in enumerate(pass_tokenize_out):
      print "tokenized chunk (%d/%d) [%d keys]" % (i+1,chunk_count, keycount)

  complete = True

  return b_dirs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-s", "--scanner", metavar='SCANNER', help="use SCANNER for tokenizing")
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--min_order", type=int, help="lowest n-gram order to use")
  parser.add_argument("--max_order", type=int, help="highest n-gram order to use")
  parser.add_argument("--word", action='store_true', default=False, help="use 'word' tokenization (currently str.split)")
  parser.add_argument("--wordn", action='store_true', default=False, help="use 'word' n-gram tokenization")
  parser.add_argument("--chunksize", type=int, help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_argument("--term_freq", action='store_true', help="count term frequency (default is document frequency)")
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-o", "--output", help="write list of output buckets to OUTPUT")
  parser.add_argument("--line", action="store_true", help="treat each line in a file as a document")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")

  group = parser.add_argument_group('sampling')
  group.add_argument("--sample_size", type=int, help="size of sample for sampling-based tokenization", default=140)
  group.add_argument("--sample_count", type=int, help="number of samples for sampling-based tokenization", default=None)
  
  args = parser.parse_args()

  if args.sample_count and args.line:
    parser.error("sampling in line mode is not implemented")
  

  if args.temp:
    tmp_dir = args.temp
  else:
    tmp_dir = os.path.join(args.model, 'buckets')
  makedir(tmp_dir)

  # We generate a new directory at each invocation, otherwise we run the 
  # risk of conflicting with a previous run without warning.
  buckets_dir = tempfile.mkdtemp(suffix='tokenize',dir=tmp_dir)

  bucketlist_path = args.output if args.output else os.path.join(args.model, 'bucketlist')
  index_path = os.path.join(args.model, 'paths')

  # display paths
  print "index path:", index_path
  print "bucketlist path:", bucketlist_path
  print "buckets path:", buckets_dir

  if args.line:
  	print "treating each LINE as a document"

  with open(index_path) as f:
    reader = csv.reader(f)
    items = list(reader)

  if sum(map(bool,(args.scanner, args.wordn, args.word))) > 1:
    parser.error('can only specify one of --word, --wordn, --scanner') 

  # Tokenize
  print "will tokenize %d files" % len(items)
  if args.scanner:
    from scanner import Scanner
    tokenizer = Scanner.from_file(args.scanner)
    print "using provided scanner: ", args.scanner
  elif args.word:
    tokenizer = str.split
    print "using str.split to tokenize"
  elif args.wordn:
    min_order = args.min_order if args.min_order else MIN_NGRAM_ORDER
    max_order = args.max_order if args.max_order else MAX_NGRAM_ORDER
    tokenizer = WordNGramTokenizer(min_order,max_order)
    print "using WORD n-gram tokenizer: min_order({0}) max_order({1})".format(min_order,max_order)
  else:
    min_order = args.min_order if args.min_order else MIN_NGRAM_ORDER
    max_order = args.max_order if args.max_order else MAX_NGRAM_ORDER
    tokenizer = NGramTokenizer(min_order,max_order)
    print "using n-gram tokenizer: min_order({0}) max_order({1})".format(min_order,max_order)
  if args.term_freq:
    print "counting term frequency"
  else:
    print "counting document frequency"
  b_dirs = build_index(items, tokenizer, buckets_dir, args.buckets, args.jobs, args.chunksize, args.sample_count, args.sample_size, args.term_freq, args.line)

  # output the paths to the buckets
  with open(bucketlist_path,'w') as f:
    for d in b_dirs:
      f.write(d+'\n')

