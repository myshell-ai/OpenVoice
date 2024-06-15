#!/usr/bin/env python
"""
NBtrain.py - 
Model generator for langid.py

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
MAX_CHUNK_SIZE = 100 # maximum number of files to tokenize at once
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation

import base64, bz2, cPickle
import os, sys, argparse, csv
import array
import numpy as np
import tempfile
import marshal
import atexit, shutil
import multiprocessing as mp
import gzip
from collections import deque, defaultdict
from contextlib import closing

from common import chunk, unmarshal_iter, read_features, index, MapPool

def state_trace(text):
  """
  Returns counts of how often each state was entered
  """
  global __nm_arr
  c = defaultdict(int)
  state = 0
  for letter in map(ord,text):
    state = __nm_arr[(state << 8) + letter]
    c[state] += 1
  return c

def setup_pass_tokenize(nm_arr, output_states, tk_output, b_dirs, line_level):
  """
  Set the global next-move array used by the aho-corasick scanner
  """
  global __nm_arr, __output_states, __tk_output, __b_dirs, __line_level
  __nm_arr = nm_arr
  __output_states = output_states
  __tk_output = tk_output
  __b_dirs = b_dirs
  __line_level = line_level

def pass_tokenize(arg):
  """
  Tokenize documents and do counts for each feature
  Split this into buckets chunked over features rather than documents

  chunk_paths contains label, path pairs because we only know the
  labels per-path, but in line mode there will be multiple documents
  per path and we don't know how many those are.
  """
  global __output_states, __tk_output, __b_dirs, __line_level
  chunk_id, chunk_paths = arg
  term_freq = defaultdict(int)

  # Tokenize each document and add to a count of (doc_id, f_id) frequencies
  doc_count = 0
  labels = []
  for label, path in chunk_paths:
    with open(path) as f:
      if __line_level:
        # each line is treated as a document
        for text in f:
          count = state_trace(text)
          for state in (set(count) & __output_states):
            for f_id in __tk_output[state]:
              term_freq[doc_count, f_id] += count[state]
          doc_count += 1
          labels.append(label)

      else:
        text = f.read()
        count = state_trace(text)
        for state in (set(count) & __output_states):
          for f_id in __tk_output[state]:
            term_freq[doc_count, f_id] += count[state]
        doc_count += 1
        labels.append(label)

  # Distribute the aggregated counts into buckets
  __procname = mp.current_process().name
  __buckets = [gzip.open(os.path.join(p,__procname+'.index'), 'a') for p in __b_dirs]
  bucket_count = len(__buckets)
  for doc_id, f_id in term_freq:
    bucket_index = hash(f_id) % bucket_count
    count = term_freq[doc_id, f_id]
    item = ( f_id, chunk_id, doc_id, count )
    __buckets[bucket_index].write(marshal.dumps(item))

  for f in __buckets:
    f.close()

  return chunk_id, doc_count, len(term_freq), labels

def setup_pass_ptc(cm, num_instances, chunk_offsets):
  global __cm, __num_instances, __chunk_offsets
  __cm = cm
  __num_instances = num_instances
  __chunk_offsets = chunk_offsets

def pass_ptc(b_dir):
  """
  Take a bucket, form a feature map, compute the count of
  each feature in each class.
  @param b_dir path to the bucket directory
  @returns (read_count, f_ids, prod) 
  """
  global __cm, __num_instances, __chunk_offsets

  terms = defaultdict(lambda : np.zeros((__num_instances,), dtype='int'))

  read_count = 0
  for path in os.listdir(b_dir):
    if path.endswith('.index'):
      for f_id, chunk_id, doc_id, count in unmarshal_iter(os.path.join(b_dir, path)):
        index = doc_id + __chunk_offsets[chunk_id]
        terms[f_id][index] = count
        read_count += 1

  f_ids, f_vs = zip(*terms.items())
  fm = np.vstack(f_vs)
  # The calculation of the term-class distribution is done per-chunk rather
  # than globally for memory efficiency reasons.
  prod = np.dot(fm, __cm)

  return read_count, f_ids, prod

def learn_nb_params(items, num_langs, tk_nextmove, tk_output, temp_path, args):
  """
  @param items label, path pairs
  """
  global outdir

  print "learning NB parameters on {} items".format(len(items))

  # Generate the feature map
  nm_arr = mp.Array('i', tk_nextmove, lock=False)

  if args.jobs:
    tasks = args.jobs * 2
  else:
    tasks = mp.cpu_count() * 2

  # Ensure chunksize of at least 1, but not exceeding specified chunksize
  chunksize = max(1, min(len(items) / tasks, args.chunksize))

  outdir = tempfile.mkdtemp(prefix="NBtrain-",suffix='-buckets', dir=temp_path)
  b_dirs = [ os.path.join(outdir,"bucket{0}".format(i)) for i in range(args.buckets) ]

  for d in b_dirs:
    os.mkdir(d)

  output_states = set(tk_output)
  
  # Divide all the items to be processed into chunks, and enumerate each chunk.
  item_chunks = list(chunk(items, chunksize))
  num_chunks = len(item_chunks)
  print "about to tokenize {} chunks".format(num_chunks)
  
  pass_tokenize_arg = enumerate(item_chunks)
  pass_tokenize_params = (nm_arr, output_states, tk_output, b_dirs, args.line) 
  with MapPool(args.jobs, setup_pass_tokenize, pass_tokenize_params) as f:
    pass_tokenize_out = f(pass_tokenize, pass_tokenize_arg)
  
    write_count = 0
    chunk_sizes = {}
    chunk_labels = []
    for i, (chunk_id, doc_count, writes, labels) in enumerate(pass_tokenize_out):
      write_count += writes
      chunk_sizes[chunk_id] = doc_count
      chunk_labels.append((chunk_id, labels))
      print "processed chunk ID:{0} ({1}/{2}) [{3} keys]".format(chunk_id, i+1, num_chunks, writes)

  print "wrote a total of %d keys" % write_count

  num_instances = sum(chunk_sizes.values())
  print "processed a total of %d instances" % num_instances

  chunk_offsets = {}
  for i in range(len(chunk_sizes)):
    chunk_offsets[i] = sum(chunk_sizes[x] for x in range(i))

  # Build CM based on re-ordeing chunk
  cm = np.zeros((num_instances, num_langs), dtype='bool')
  for chunk_id, chunk_label in chunk_labels:
    for doc_id, lang_id in enumerate(chunk_label):
      index = doc_id + chunk_offsets[chunk_id]
      cm[index, lang_id] = True

  pass_ptc_params = (cm, num_instances, chunk_offsets)
  with MapPool(args.jobs, setup_pass_ptc, pass_ptc_params) as f:
    pass_ptc_out = f(pass_ptc, b_dirs)

    def pass_ptc_progress():
      for i,v in enumerate(pass_ptc_out):
        yield v
        print "processed chunk ({0}/{1})".format(i+1, len(b_dirs))

    reads, ids, prods = zip(*pass_ptc_progress())
    read_count = sum(reads)
    print "read a total of %d keys (%d short)" % (read_count, write_count - read_count)

  num_features = max( i for v in tk_output.values() for i in v) + 1
  prod = np.zeros((num_features, cm.shape[1]), dtype=int)
  prod[np.concatenate(ids)] = np.vstack(prods)

  # This is where the smoothing occurs
  ptc = np.log(1 + prod) - np.log(num_features + prod.sum(0))

  nb_ptc = array.array('d')
  for term_dist in ptc.tolist():
    nb_ptc.extend(term_dist)

  pc = np.log(cm.sum(0))
  nb_pc = array.array('d', pc)

  return nb_pc, nb_ptc

@atexit.register
def cleanup():
  global outdir 
  try:
    shutil.rmtree(outdir)
  except NameError:
    pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-s", "--scanner", metavar='SCANNER', help="use SCANNER for feature counting")
  parser.add_argument("-o", "--output", metavar='OUTPUT', help="output langid.py-compatible model to OUTPUT")
  #parser.add_argument("-i","--index",metavar='INDEX',help="read list of training document paths from INDEX")
  parser.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")
  parser.add_argument("--chunksize", type=int, help='maximum chunk size (number of files)', default=MAX_CHUNK_SIZE)
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--line", action="store_true", help="treat each line in a file as a document")
  args = parser.parse_args()

  if args.temp:
    temp_path = args.temp
  else:
    temp_path = os.path.join(args.model, 'buckets')

  if args.scanner:
    scanner_path = args.scanner
  else:
    scanner_path = os.path.join(args.model, 'LDfeats.scanner')

  if args.output:
    output_path = args.output
  else:
    output_path = os.path.join(args.model, 'model')

  index_path = os.path.join(args.model, 'paths')
  lang_path = os.path.join(args.model, 'lang_index')

  # display paths
  print "model path:", args.model
  print "temp path:", temp_path
  print "scanner path:", scanner_path
  print "output path:", output_path

  if args.line:
    print "treating each LINE as a document"

  # read list of training files
  with open(index_path) as f:
    reader = csv.reader(f)
    items = [ (int(l),p) for _,l,p in reader ]

  # read scanner
  with open(scanner_path) as f:
    tk_nextmove, tk_output, _ = cPickle.load(f)

  # read list of languages in order
  with open(lang_path) as f:
    reader = csv.reader(f)
    langs = zip(*reader)[0]
    
  nb_classes = langs
  nb_pc, nb_ptc = learn_nb_params(items, len(langs), tk_nextmove, tk_output, temp_path, args)

  # output the model
  model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  print "wrote model to %s (%d bytes)" % (output_path, len(string))
