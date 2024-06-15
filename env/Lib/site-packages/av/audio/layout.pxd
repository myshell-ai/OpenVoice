from libc.stdint cimport uint64_t


cdef class AudioLayout(object):

    # The layout for FFMpeg; this is essentially a bitmask of channels.
    cdef uint64_t layout
    cdef int nb_channels

    cdef readonly tuple channels
    """
    A tuple of :class:`AudioChannel` objects.

    :type: tuple
    """

    cdef _init(self, uint64_t layout)


cdef class AudioChannel(object):

    # The channel for FFmpeg.
    cdef uint64_t channel


cdef AudioLayout get_audio_layout(int channels, uint64_t c_layout)
