# -*- coding: utf-8 -*-

"""Console script for dtw."""
import sys
import numpy
import dtw
import argparse


def main2(query, reference, step_pattern):
    """Console script for dtw."""

    q = numpy.genfromtxt(query)
    r = numpy.genfromtxt(reference)
    al = dtw.dtw(q, r, step_pattern=step_pattern)

    wp = numpy.vstack([al.index1, al.index2])

    out = ""
    if hasattr(al,"normalizedDistance"):
        out += f"Normalized distance: {al.normalizedDistance:.4g}\n\n"

    out += f"Distance: {al.distance:.4g}\n\n"
    out += f"Warping path: {wp}\n\n"

    return out

def main():
    parser = argparse.ArgumentParser(description='Command line DTW utility.',
                                     epilog="\nThe Python and R interfaces provide the full functionality, including plots.\n"+\
                                     "See https://dynamictimewarping.github.io/\n\n")
    parser.add_argument("query",  help="Query timeseries (tsv)")
    parser.add_argument("reference",  help="Reference timeseries (tsv)")
    parser.add_argument("--step_pattern", default="symmetric2", help="Step pattern, aka recursion rule. E.g. symmetric2, asymmetric, ...")

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    opts = parser.parse_args()
    out=main2(opts.query, opts.reference, opts.step_pattern)
    
    print(out)
    

if __name__ == "__main__":
    main()

