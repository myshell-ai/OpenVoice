"""
Tabulate feature weight data into a single CSV for
further analysis using other tools. This produces
a CSV with header. The features themselves are not
included.

Marco Lui, February 2013
"""

import argparse, os, csv, sys
import numpy as np
import bz2, base64
from cPickle import loads

from langid.train.common import read_weights, read_features

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('model', metavar="MODEL_DIR", help="path to langid.py training model dir")
  parser.add_argument('output', metavar="OUTPUT", help = "write to OUTPUT")
  parser.add_argument('-f','--features', metavar="FILE", help = 'only output features from FILE')
  parser.add_argument('--raw', action='store_true', help="include raw features")
  parser.add_argument('--bin', action='store_true', help="include ig for lang-bin")
  args = parser.parse_args()

  def model_file(name):
    return os.path.join(args.model, name)

  # Try to determine the set of features to consider
  if args.features:
    # Use a pre-determined feature list
    print >>sys.stderr,  "using user-supplied feature list:", args.features
    feats = read_features(args.features)
  elif os.path.exists(model_file('LDfeats')):
    # Use LDfeats
    print >>sys.stderr,  "using LDfeats"
    feats = read_features(model_file('LDfeats'))
  else:
    raise ValueError("no suitable feature list")

  print >>sys.stderr, "considering {0} features".format(len(feats))

  records = dict( (k, {}) for k in feats )
  headers = []

  headers.append('len')
  for k in feats:
    records[k]['len'] = len(k)


  # Document Frequency
  if os.path.exists(model_file('DF_all')):
    print >>sys.stderr, "found weights for document frequency"
    w = read_weights(model_file('DF_all'))
    headers.append('DF')
    for k in feats:
      records[k]['DF'] = w[k][0]

  # IG weights for the all-languages event
  if os.path.exists(model_file('IGweights.lang')):
    print >>sys.stderr, "found weights for lang"
    w = read_weights(model_file('IGweights.lang'))
    headers.append('IGlang')
    for k in feats:
      records[k]['IGlang'] = w[k][0]

  # IG weights for the all-domains event
  if os.path.exists(model_file('IGweights.domain')):
    print >>sys.stderr, "found weights for domain"
    w = read_weights(model_file('IGweights.domain'))
    headers.append('IGdomain')
    for k in feats:
      records[k]['IGdomain'] = w[k][0]

  # IG weights for language-binarized
  if args.bin and os.path.exists(model_file('IGweights.lang.bin')) and os.path.exists(model_file('lang_index')):
    print >>sys.stderr, "found weights for lang.bin"
    w = read_weights(model_file('IGweights.lang.bin'))

    # find the list of langs in-order
    with open(os.path.join(args.model, "lang_index")) as f:
      reader = csv.reader(f)
      langs = zip(*reader)[0]

    r_h = ['IGlang.bin.{0}'.format(l) for l in langs]
    headers.extend( r_h )
    for k in feats:
      records[k].update( dict(zip(r_h, w[k])) )
        
  if os.path.exists(model_file('LDfeats.scanner')) and os.path.exists(model_file('model')):
    print >>sys.stderr, "found weights for P(t|c)"
    with open(model_file('model')) as f:
      model = loads(bz2.decompress(base64.b64decode(f.read())))
    with open(model_file('LDfeats.scanner')) as f:
      _, _, nb_feats = loads(f.read())
    nb_ptc, nb_pc, nb_classes, tk_nextmove, tk_output = model
    nb_numfeats = len(nb_ptc) / len(nb_pc)
    nb_ptc = np.array(nb_ptc).reshape(len(nb_ptc)/len(nb_pc), len(nb_pc))

    # Normalize to 1 on the term axis
    for i in range(nb_ptc.shape[1]):
      nb_ptc[:,i] = (1/np.exp(nb_ptc[:,i][None,:] - nb_ptc[:,i][:,None]).sum(1))
    w = dict(zip(nb_feats, nb_ptc))

    r_h = ['ptc.{0}'.format(l) for l in nb_classes]
    headers.extend( r_h )
    for k in feats:
      records[k].update( dict(zip(r_h, w[k])) )

  if args.raw:
    headers.append('feat')
    for k in feats:
      records[k]['feat'] = k



  print >>sys.stderr, "writing output"
  with open(args.output, 'w') as f:
    writer = csv.DictWriter(f,headers)
    writer.writeheader()
    writer.writerows(records.values())
  
  print >>sys.stderr, "done"
