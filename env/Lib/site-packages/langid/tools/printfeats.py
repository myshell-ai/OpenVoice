"""
Print features out in order of their weights

Marco Lui, November 2013
"""

import argparse, os, csv, sys

from langid.train.common import read_weights

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('file', help="file to read")
  parser.add_argument('-c','--column',help="project a specific column", type=int)
  parser.add_argument('-n','--number',help="output top N features", type=int)
  parser.add_argument('-v','--value',help="output the value used for ranking", action="store_true")
  parser.add_argument('-p','--printfeat',help="print the actual feature (default is to print repr)", action="store_true")
  parser.add_argument('--output', "-o", default=sys.stdout, type=argparse.FileType('w'), help = "write to OUTPUT")
  args = parser.parse_args()

  w = read_weights(args.file)
  n = args.number if args.number is not None else len(w)

  def show(feat):
    if args.printfeat:
      return feat
    else:
      return repr(feat)

  if args.column is not None:
    for key in sorted(w, key=lambda x:w[x][args.column], reverse=True)[:n]:
      if args.value:
        args.output.write("{0},{1}\n".format(show(key),w[key][args.column]))
      else:
        args.output.write("{0}\n".format(show(key)))
  else:
    for key in sorted(w, key=w.get, reverse=True)[:n]:
      if args.value:
        args.output.write("{0},{1}\n".format(show(key),w[key]))
      else:
        args.output.write("{0}\n".format(show(key)))

