cimport libav as lib

from av.filter.context cimport FilterContext


cdef class Graph(object):

    cdef lib.AVFilterGraph *ptr

    cdef readonly bint configured
    cpdef configure(self, bint auto_buffer=*, bint force=*)

    cdef dict _name_counts
    cdef str _get_unique_name(self, str name)

    cdef _register_context(self, FilterContext)
    cdef _auto_register(self)
    cdef int _nb_filters_seen
    cdef dict _context_by_ptr
    cdef dict _context_by_name
    cdef dict _context_by_type
