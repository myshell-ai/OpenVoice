from cpython.buffer cimport Py_buffer


cdef class ByteSource(object):

    cdef object owner

    cdef bint has_view
    cdef Py_buffer view

    cdef unsigned char *ptr
    cdef size_t length

cdef ByteSource bytesource(object, bint allow_none=*)
