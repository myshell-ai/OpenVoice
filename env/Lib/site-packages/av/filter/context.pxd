cimport libav as lib

from av.filter.filter cimport Filter
from av.filter.graph cimport Graph


cdef class FilterContext(object):

    cdef lib.AVFilterContext *ptr
    cdef readonly Graph graph
    cdef readonly Filter filter

    cdef object _inputs
    cdef object _outputs

    cdef bint inited


cdef FilterContext wrap_filter_context(Graph graph, Filter filter, lib.AVFilterContext *ptr)
