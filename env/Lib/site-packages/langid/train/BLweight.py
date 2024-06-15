"""
Implementing the "blacklist" feature weighting metric proposed by
Tiedemann & Ljubesic.

Marco Lui, February 2013
"""

NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)

import os
import argparse
import numpy as np

from common import read_features, makedir, write_weights
from scanner import build_scanner
from index import CorpusIndexer
from NBtrain import generate_cm, learn_pc, learn_ptc


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("-o","--output", metavar="DIR", help = "write weights to DIR")
  parser.add_argument('-f','--features', metavar="FILE", help = 'only output features from FILE')
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', help="store buckets in TEMP_DIR instead of in MODEL_DIR/buckets")
  parser.add_argument("-j","--jobs", type=int, metavar='N', help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
  parser.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  parser.add_argument("--chunksize", type=int, help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)", default=CHUNKSIZE)
  parser.add_argument("--no_norm", default=False, action="store_true", help="do not normalize difference in p(t|C) by sum p(t|C)")
  parser.add_argument("corpus", help="read corpus from CORPUS_DIR", metavar="CORPUS_DIR")
  parser.add_argument("pairs", metavar='LANG_PAIR', nargs="*", help="language pairs to compute BL weights for")
  args = parser.parse_args()

  # Work out where our model directory is
  corpus_name = os.path.basename(args.corpus)
  if args.model:
    model_dir = args.model
  else:
    model_dir = os.path.join('.', corpus_name+'.model')

  def m_path(name):
    return os.path.join(model_dir, name)

  # Try to determine the set of features to consider
  if args.features:
    # Use a pre-determined feature list
    feat_path = args.features
  elif os.path.exists(m_path('DFfeats')):
    # Use LDfeats
    feat_path = m_path('DFfeats')
  else:
    raise ValueError("no suitable feature list")

  # Where temp files go
  if args.temp:
    buckets_dir = args.temp
  else:
    buckets_dir = m_path('buckets')
  makedir(buckets_dir)

  all_langs = set()
  pairs = []
  for p in args.pairs:
    try:
      lang1, lang2 = p.split(',')
    except ValueError:
      # Did not unpack to two values
      parser.error("{0} is not a lang-pair".format(p))
    all_langs.add(lang1)
    all_langs.add(lang2)
    pairs.append((lang1, lang2))

  if args.output:
    makedir(args.output)
    out_dir = args.output
  else:
    out_dir = model_dir

  langs = sorted(all_langs)

  # display paths
  print "languages({1}): {0}".format(langs, len(langs))
  print "model path:", model_dir
  print "feature path:", feat_path
  print "output path:", out_dir
  print "temp (buckets) path:", buckets_dir

  feats = read_features(feat_path)

  indexer = CorpusIndexer(args.corpus, langs = langs)
  items = [ (d,l,p) for (d,l,n,p) in indexer.items ]
  if len(items) == 0:
    raise ValueError("found no files!")

  print "will process {0} features across {1} paths".format(len(feats), len(items))

  # produce a scanner over all the features
  tk_nextmove, tk_output = build_scanner(feats)

  # Generate a class map over all the languages we are dealing with
  cm = generate_cm([ (l,p) for d,l,p in items], len(langs))

  # Compute P(t|C)
  print "learning P(t|C)"
  paths = zip(*items)[2]
  nb_ptc = learn_ptc(paths, tk_nextmove, tk_output, cm, buckets_dir, args)
  nb_ptc = np.array(nb_ptc).reshape(len(feats), len(langs))

  # Normalize to 1 on the term axis
  print "renormalizing P(t|C)"
  for i in range(nb_ptc.shape[1]):
    # had to de-vectorize this due to memory consumption
    newval = np.empty_like(nb_ptc[:,i])
    for j in range(newval.shape[0]):
      newval[j] = (1/np.exp(nb_ptc[:,i] - nb_ptc[j,i]).sum())
    nb_ptc[:,i] = newval
    assert (1.0 - newval.sum()) < 0.0001

  print "doing per-pair output"
  for lang1, lang2 in pairs:
    # Where to do output
    if args.no_norm:
      weights_path = os.path.join(out_dir, ('BLfeats.no_norm.{0}.{1}'.format(lang1, lang2)))
    else:
      weights_path = os.path.join(out_dir, ('BLfeats.{0}.{1}'.format(lang1, lang2)))

    i1 = indexer.lang_index[lang1]
    i2 = indexer.lang_index[lang2]

    w = dict(zip(feats, np.abs((nb_ptc[:,i1] - nb_ptc[:,i2]) / (nb_ptc.sum(1) if not args.no_norm else 1))))
    write_weights(w, weights_path)
    print "wrote weights to {0}".format(weights_path)
