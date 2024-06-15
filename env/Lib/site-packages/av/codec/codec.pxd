cimport libav as lib


cdef class Codec(object):

    cdef const lib.AVCodec *ptr
    cdef const lib.AVCodecDescriptor *desc
    cdef readonly bint is_encoder

    cdef _init(self, name=?)


cdef Codec wrap_codec(const lib.AVCodec *ptr)
