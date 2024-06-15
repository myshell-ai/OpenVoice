cimport libav as lib

from av.container.pyio cimport PyIOFile
from av.container.streams cimport StreamContainer
from av.dictionary cimport _Dictionary
from av.format cimport ContainerFormat
from av.stream cimport Stream


# Interrupt callback information, times are in seconds.
ctypedef struct timeout_info:
    double start_time
    double timeout


cdef class Container(object):

    cdef readonly bint writeable
    cdef lib.AVFormatContext *ptr

    cdef readonly object name
    cdef readonly str metadata_encoding
    cdef readonly str metadata_errors

    cdef readonly PyIOFile file
    cdef int buffer_size
    cdef bint input_was_opened
    cdef readonly object io_open
    cdef readonly object open_files

    cdef readonly ContainerFormat format

    cdef readonly dict options
    cdef readonly dict container_options
    cdef readonly list stream_options

    cdef readonly StreamContainer streams
    cdef readonly dict metadata

    cdef int err_check(self, int value) except -1

    # Timeouts
    cdef readonly object open_timeout
    cdef readonly object read_timeout
    cdef timeout_info interrupt_callback_info
    cdef set_timeout(self, object)
    cdef start_timeout(self)
