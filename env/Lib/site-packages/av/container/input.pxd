cimport libav as lib

from av.container.core cimport Container
from av.stream cimport Stream


cdef class InputContainer(Container):

    cdef flush_buffers(self)
