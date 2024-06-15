#!/usr/bin/env python
"""
train.py - 
All-in-one tool for easy training of a model for langid.py. This depends on the
training tools for individual steps, which can be run separately.

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

TRAIN_PROP = 1.0 # probability than any given document is selected
MIN_DOMAIN = 1 # minimum number of domains a language must be present in to be included
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOP_DOC_FREQ = 15000 # number of tokens to consider for each order
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)
FEATURES_PER_LANG = 300 # number of features to select for each language

import argparse
import os, csv
import numpy
import base64, bz2, cPickle
import shutil

from common import makedir, write_weights, write_features, read_weights, read_features
from index import CorpusIndexer
from tokenize import build_index, NGramTokenizer
from DFfeatureselect import tally, ngram_select
from IGweight import compute_IG
from LDfeatureselect import select_LD_features
from scanner import build_scanner, Scanner
from NBtrain import learn_nb_params

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-p","--proportion", type=float, help="proportion of training data to use", default=TRAIN_PROP)
  parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-d","--domain", metavar="DOMAIN", action='append',
      help="use DOMAIN - can be specified multiple times (uses all domains found if not specified)")
  parser.add_argument("-l","--lang", metavar="LANG", action='append',
      help="use LANG - can be specified multiple times (uses all langs found if not specified)")
  parser.add_argument("--min_domain", type=int, help="minimum number of domains a language must be present in", default=MIN_DOMAIN)
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--max_order", type=int, help="highest n-gram order to use", default=MAX_NGRAM_ORDER)
  parser.add_argument("--chunksize", type=int, help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_argument("--df_tokens", type=int, help="number of tokens to consider for each n-gram order", default=TOP_DOC_FREQ)
  parser.add_argument("--word", action='store_true', default=False, help="use 'word' tokenization (currently str.split)")
  parser.add_argument("--df_feats", metavar="FEATS", help="Instead of DF feature selection, use a list of features from FEATS")
  parser.add_argument("--ld_feats", metavar="FEATS", help="Instead of LD feature selection, use a list of features from FEATS")
  parser.add_argument("--feats_per_lang", type=int, metavar='N', help="select top N features for each language", default=FEATURES_PER_LANG)
  parser.add_argument("--no_domain_ig", action="store_true", default=False, help="use only per-langugage IG in LD calculation")
  parser.add_argument("--debug", action="store_true", default=False, help="produce debug output (all intermediates)")
  parser.add_argument("--line", action="store_true", help="treat each line in a file as a document")

  group = parser.add_argument_group('sampling')
  group.add_argument("--sample_size", type=int, help="size of sample for sampling-based tokenization", default=140)
  group.add_argument("--sample_count", type=int, help="number of samples for sampling-based tokenization", default=None)

  parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")

  args = parser.parse_args()

  if args.sample_count and args.line:
    parser.error("sampling in line mode is not implemented")

  if args.df_feats and args.ld_feats:
    parser.error("--df_feats and --ld_feats are mutually exclusive")

  corpus_name = os.path.basename(args.corpus)
  if args.model:
    model_dir = args.model
  else:
    model_dir = os.path.join('.', corpus_name+'.model')

  makedir(model_dir)

  # display paths
  print "corpus path:", args.corpus
  print "model path:", model_dir

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

  if args.line:
  	print "treating each LINE as a document"

  items = sorted(set( (d,l,p) for (d,l,n,p) in indexer.items ))
  if args.debug:
    langs_path = os.path.join(model_dir, 'lang_index')
    domains_path = os.path.join(model_dir, 'domain_index')
    index_path = os.path.join(model_dir, 'paths')

    # output the language index
    with open(langs_path,'w') as f:
      writer = csv.writer(f)
      writer.writerows((l, lang_dist[lang_index[l]]) 
          for l in sorted(lang_index, key=lang_index.get))

    # output the domain index
    with open(domains_path,'w') as f:
      writer = csv.writer(f)
      writer.writerows((d, domain_dist[domain_index[d]]) 
          for d in sorted(domain_index, key=domain_index.get))

    # output items found
    with open(index_path,'w') as f:
      writer = csv.writer(f)
      writer.writerows(items)

  if args.temp:
    buckets_dir = args.temp
  else:
    buckets_dir = os.path.join(model_dir, 'buckets')
  makedir(buckets_dir)


  if args.ld_feats:
    # LD features are pre-specified. We are basically just building the NB model.
    LDfeats = read_features(args.ld_feats)

  else:
    # LD features not pre-specified, so we compute them.

    # Tokenize
    DFfeats = None
    print "will tokenize %d documents" % len(items)
    # TODO: Custom tokenizer if doing custom first-pass features
    if args.df_feats:
      print "reading custom features from:", args.df_feats
      DFfeats = read_features(args.df_feats)
      print "building tokenizer for custom list of {0} features".format(len(DFfeats))
      tk = Scanner(DFfeats)
    elif args.word:
      print "using word tokenizer"
      tk = str.split
    else:
      print "using byte NGram tokenizer, max_order: {0}".format(args.max_order)
      tk = NGramTokenizer(1, args.max_order)
    
    # First-pass tokenization, used to determine DF of features
    tk_dir = os.path.join(buckets_dir, 'tokenize-pass1')
    makedir(tk_dir)
    b_dirs = build_index(items, tk, tk_dir, args.buckets, args.jobs, args.chunksize, args.sample_count, args.sample_size, args.line)

    if args.debug:
      # output the paths to the buckets
      bucketlist_path = os.path.join(model_dir, 'bucketlist')
      with open(bucketlist_path,'w') as f:
        for d in b_dirs:
          f.write(d+'\n')

    # We need to compute a tally if we are selecting features by DF, but also if
    # we want full debug output.
    if DFfeats is None or args.debug:
      # Compute DF per-feature
      doc_count = tally(b_dirs, args.jobs)
      if args.debug:
        doc_count_path = os.path.join(model_dir, 'DF_all')
        write_weights(doc_count, doc_count_path)
        print "wrote DF counts for all features to:", doc_count_path

    if DFfeats is None:
      # Choose the first-stage features
      DFfeats = ngram_select(doc_count, args.max_order, args.df_tokens)

    if args.debug:
      feature_path = os.path.join(model_dir, 'DFfeats')
      write_features(DFfeats, feature_path)
      print 'wrote features to "%s"' % feature_path 

    # Dispose of the first-pass tokenize output as it is no longer 
    # needed.
    if not args.debug:
      shutil.rmtree(tk_dir)

    # Second-pass tokenization to only obtain counts for the selected features.
    # As the first-pass set is typically much larger than the second pass, it often 
    # works out to be faster to retokenize the raw documents rather than iterate
    # over the first-pass counts.
    DF_scanner = Scanner(DFfeats)
    df_dir = os.path.join(buckets_dir, 'tokenize-pass2')
    makedir(df_dir)
    b_dirs = build_index(items, DF_scanner, df_dir, args.buckets, args.jobs, args.chunksize)
    b_dirs = [[d] for d in b_dirs]

    # Build vectors of domain and language distributions for use in IG calculation
    domain_dist_vec = numpy.array([ domain_dist[domain_index[d]]
            for d in sorted(domain_index, key=domain_index.get)], dtype=int)
    lang_dist_vec = numpy.array([ lang_dist[lang_index[l]]
            for l in sorted(lang_index.keys(), key=lang_index.get)], dtype=int)

    # Compute IG
    ig_params = [
      ('lang', lang_dist_vec, '.lang', True),
    ]
    if not args.no_domain_ig:
      ig_params.append( ('domain', domain_dist_vec, '.domain', False) )

    ig_vals = {}
    for label, dist, suffix, binarize in ig_params:
      print "Computing information gain for {0}".format(label)
      ig = compute_IG(b_dirs, DFfeats, dist, binarize, suffix, args.jobs)
      if args.debug:
        weights_path = os.path.join(model_dir, 'IGweights' + suffix + ('.bin' if binarize else ''))
        write_weights(ig, weights_path)
      ig_vals[label] = dict((row[0], numpy.array(row[1].flat)) for row in ig)

    # Select features according to the LD criteria
    features_per_lang = select_LD_features(ig_vals['lang'], ig_vals.get('domain'), args.feats_per_lang, ignore_domain = args.no_domain_ig)
    LDfeats = reduce(set.union, map(set, features_per_lang.values()))
    print 'selected %d features' % len(LDfeats)

    if args.debug:
      feature_path = os.path.join(model_dir, 'LDfeats')
      write_features(sorted(LDfeats), feature_path)
      print 'wrote LD features to "%s"' % feature_path 

      with open(feature_path + '.perlang', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(features_per_lang)):
          writer.writerow(map(repr,features_per_lang[i]))
      print 'wrote LD.perlang features to "%s"' % feature_path + '.perlang'

  # Compile a scanner for the LDfeats
  tk_nextmove, tk_output = build_scanner(LDfeats)
  if args.debug:
    scanner_path = feature_path + '.scanner'
    with open(scanner_path, 'w') as f:
      cPickle.dump((tk_nextmove, tk_output, LDfeats), f)
    print "wrote scanner to {0}".format(scanner_path)

  # Assemble the NB model
  langs = sorted(lang_index, key=lang_index.get)

  nb_classes = langs
  nb_dir = os.path.join(buckets_dir, 'NBtrain')
  makedir(nb_dir)
  nb_pc, nb_ptc = learn_nb_params([(int(l),p) for _, l, p in items], len(langs), tk_nextmove, tk_output, nb_dir, args)

  # output the model
  output_path = os.path.join(model_dir, 'model')
  model = nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  print "wrote model to %s (%d bytes)" % (output_path, len(string))

  # remove buckets if debug is off. We don't generate buckets if ldfeats is supplied.
  if not args.debug and not args.ld_feats:
    shutil.rmtree(df_dir)
    if not args.temp:
      # Do not remove the buckets dir if temp was supplied as we don't know
      # if we created it.
      shutil.rmtree(buckets_dir)
    

