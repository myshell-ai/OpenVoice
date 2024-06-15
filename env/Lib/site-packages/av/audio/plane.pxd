from av.plane cimport Plane


cdef class AudioPlane(Plane):

    cdef readonly size_t buffer_size

    cdef size_t _buffer_size(self)
