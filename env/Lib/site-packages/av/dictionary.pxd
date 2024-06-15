cimport libav as lib


cdef class _Dictionary(object):

    cdef lib.AVDictionary *ptr

    cpdef _Dictionary copy(self)


cdef _Dictionary wrap_dictionary(lib.AVDictionary *input_)
