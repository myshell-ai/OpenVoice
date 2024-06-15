cimport libav as lib


cdef class ContainerFormat(object):

    cdef readonly str name

    cdef lib.AVInputFormat *iptr
    cdef lib.AVOutputFormat *optr


cdef ContainerFormat build_container_format(lib.AVInputFormat*, lib.AVOutputFormat*)
