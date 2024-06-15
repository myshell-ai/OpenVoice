cimport libav as lib

from av.frame cimport Frame
from av.sidedata.sidedata cimport SideData


cdef class _MotionVectors(SideData):

    cdef dict _vectors
    cdef int _len


cdef class MotionVector(object):

    cdef _MotionVectors parent
    cdef lib.AVMotionVector *ptr
