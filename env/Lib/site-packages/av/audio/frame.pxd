from libc.stdint cimport uint8_t, uint64_t
cimport libav as lib

from av.audio.format cimport AudioFormat
from av.audio.layout cimport AudioLayout
from av.frame cimport Frame


cdef class AudioFrame(Frame):

    # For raw storage of the frame's data; don't ever touch this.
    cdef uint8_t *_buffer
    cdef size_t _buffer_size

    cdef readonly AudioLayout layout
    """
    The audio channel layout.

    :type: AudioLayout
    """

    cdef readonly AudioFormat format
    """
    The audio sample format.

    :type: AudioFormat
    """

    cdef _init(self, lib.AVSampleFormat format, uint64_t layout, unsigned int nb_samples, unsigned int align)
    cdef _init_user_attributes(self)

cdef AudioFrame alloc_audio_frame()
