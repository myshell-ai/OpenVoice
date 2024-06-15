from av.stream cimport Stream


cdef class StreamContainer(object):

    cdef list _streams

    # For the different types.
    cdef readonly tuple video
    cdef readonly tuple audio
    cdef readonly tuple subtitles
    cdef readonly tuple data
    cdef readonly tuple other

    cdef add_stream(self, Stream stream)
