# This file is part of audioread.
# Copyright 2011, Adrian Sampson.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""Use Gstreamer to decode audio files.

To read an audio file, pass it to the constructor for GstAudioFile()
and then iterate over the contents:

    >>> f = GstAudioFile('something.mp3')
    >>> try:
    >>>     for block in f:
    >>>         ...
    >>> finally:
    >>>     f.close()

Note that there are a few complications caused by Gstreamer's
asynchronous architecture. This module spawns its own Gobject main-
loop thread; I'm not sure how that will interact with other main
loops if your program has them. Also, in order to stop the thread
and terminate your program normally, you need to call the close()
method on every GstAudioFile you create. Conveniently, the file can be
used as a context manager to make this simpler:

    >>> with GstAudioFile('something.mp3') as f:
    >>>     for block in f:
    >>>         ...

Iterating a GstAudioFile yields strings containing short integer PCM
data. You can also read the sample rate and channel count from the
file:

    >>> with GstAudioFile('something.mp3') as f:
    >>>     print f.samplerate
    >>>     print f.channels
    >>>     print f.duration
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import sys
import threading
import os
import queue
from urllib.parse import quote

from .exceptions import DecodeError
from .base import AudioFile

QUEUE_SIZE = 10
BUFFER_SIZE = 10
SENTINEL = '__GSTDEC_SENTINEL__'


# Exceptions.

class GStreamerError(DecodeError):
    pass


class UnknownTypeError(GStreamerError):
    """Raised when Gstreamer can't decode the given file type."""
    def __init__(self, streaminfo):
        super().__init__(
            "can't decode stream: " + streaminfo
        )
        self.streaminfo = streaminfo


class FileReadError(GStreamerError):
    """Raised when the file can't be read at all."""
    pass


class NoStreamError(GStreamerError):
    """Raised when the file was read successfully but no audio streams
    were found.
    """
    def __init__(self):
        super().__init__('no audio streams found')


class MetadataMissingError(GStreamerError):
    """Raised when GStreamer fails to report stream metadata (duration,
    channels, or sample rate).
    """
    pass


class IncompleteGStreamerError(GStreamerError):
    """Raised when necessary components of GStreamer (namely, the
    principal plugin packages) are missing.
    """
    def __init__(self):
        super().__init__(
            'missing GStreamer base plugins'
        )


# Managing the Gobject main loop thread.

_shared_loop_thread = None
_loop_thread_lock = threading.RLock()

Gst.init(None)

def get_loop_thread():
    """Get the shared main-loop thread.
    """
    global _shared_loop_thread
    with _loop_thread_lock:
        if not _shared_loop_thread:
            # Start a new thread.
            _shared_loop_thread = MainLoopThread()
            _shared_loop_thread.start()
        return _shared_loop_thread


class MainLoopThread(threading.Thread):
    """A daemon thread encapsulating a Gobject main loop.
    """
    def __init__(self):
        super().__init__()
        self.loop = GLib.MainLoop.new(None, False)
        self.daemon = True

    def run(self):
        self.loop.run()


# The decoder.

class GstAudioFile(AudioFile):
    """Reads raw audio data from any audio file that Gstreamer
    knows how to decode.

        >>> with GstAudioFile('something.mp3') as f:
        >>>     print f.samplerate
        >>>     print f.channels
        >>>     print f.duration
        >>>     for block in f:
        >>>         do_something(block)

    Iterating the object yields blocks of 16-bit PCM data. Three
    pieces of stream information are also available: samplerate (in Hz),
    number of channels, and duration (in seconds).

    It's very important that the client call close() when it's done
    with the object. Otherwise, the program is likely to hang on exit.
    Alternatively, of course, one can just use the file as a context
    manager, as shown above.
    """
    def __init__(self, path):
        self.running = False
        self.finished = False

        # Set up the Gstreamer pipeline.
        self.pipeline = Gst.Pipeline()

        self.dec = Gst.ElementFactory.make("uridecodebin", None)
        self.conv = Gst.ElementFactory.make("audioconvert", None)
        self.sink = Gst.ElementFactory.make("appsink", None)

        if self.dec is None or self.conv is None or self.sink is None:
            # uridecodebin, audioconvert, or appsink is missing. We need
            # gst-plugins-base.
            raise IncompleteGStreamerError()

        # Register for bus signals.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", self._message)
        bus.connect("message::error", self._message)

        # Configure the input.
        uri = 'file://' + quote(os.path.abspath(path))
        self.dec.set_property("uri", uri)
        # The callback to connect the input.
        self.dec.connect("pad-added", self._pad_added)
        self.dec.connect("no-more-pads", self._no_more_pads)
        # And a callback if decoding fails.
        self.dec.connect("unknown-type", self._unkown_type)

        # Configure the output.
        # We want short integer data.
        self.sink.set_property(
            'caps',
            Gst.Caps.from_string('audio/x-raw, format=(string)S16LE'),
        )
        # TODO set endianness?
        # Set up the characteristics of the output. We don't want to
        # drop any data (nothing is real-time here); we should bound
        # the memory usage of the internal queue; and, most
        # importantly, setting "sync" to False disables the default
        # behavior in which you consume buffers in real time. This way,
        # we get data as soon as it's decoded.
        self.sink.set_property('drop', False)
        self.sink.set_property('max-buffers', BUFFER_SIZE)
        self.sink.set_property('sync', False)
        # The callback to receive decoded data.
        self.sink.set_property('emit-signals', True)
        self.sink.connect("new-sample", self._new_sample)

        # We'll need to know when the stream becomes ready and we get
        # its attributes. This semaphore will become available when the
        # caps are received. That way, when __init__() returns, the file
        # (and its attributes) will be ready for reading.
        self.ready_sem = threading.Semaphore(0)
        self.caps_handler = self.sink.get_static_pad("sink").connect(
            "notify::caps", self._notify_caps
        )

        # Link up everything but the decoder (which must be linked only
        # when it becomes ready).
        self.pipeline.add(self.dec)
        self.pipeline.add(self.conv)
        self.pipeline.add(self.sink)

        self.conv.link(self.sink)

        # Set up the queue for data and run the main thread.
        self.queue = queue.Queue(QUEUE_SIZE)
        self.thread = get_loop_thread()

        # This wil get filled with an exception if opening fails.
        self.read_exc = None

        # Return as soon as the stream is ready!
        self.running = True
        self.got_caps = False
        self.pipeline.set_state(Gst.State.PLAYING)
        self.ready_sem.acquire()
        if self.read_exc:
            # An error occurred before the stream became ready.
            self.close(True)
            raise self.read_exc

    # Gstreamer callbacks.

    def _notify_caps(self, pad, args):
        """The callback for the sinkpad's "notify::caps" signal.
        """
        # The sink has started to receive data, so the stream is ready.
        # This also is our opportunity to read information about the
        # stream.
        self.got_caps = True
        info = pad.get_current_caps().get_structure(0)

        # Stream attributes.
        self.channels = info.get_int('channels')[1]
        self.samplerate = info.get_int('rate')[1]

        # Query duration.
        success, length = pad.get_peer().query_duration(Gst.Format.TIME)
        if success:
            self.duration = length / 1000000000
        else:
            self.read_exc = MetadataMissingError('duration not available')

        # Allow constructor to complete.
        self.ready_sem.release()

    _got_a_pad = False

    def _pad_added(self, element, pad):
        """The callback for GstElement's "pad-added" signal.
        """
        # Decoded data is ready. Connect up the decoder, finally.
        name = pad.query_caps(None).to_string()
        if name.startswith('audio/x-raw'):
            nextpad = self.conv.get_static_pad('sink')
            if not nextpad.is_linked():
                self._got_a_pad = True
                pad.link(nextpad)

    def _no_more_pads(self, element):
        """The callback for GstElement's "no-more-pads" signal.
        """
        # Sent when the pads are done adding (i.e., there are no more
        # streams in the file). If we haven't gotten at least one
        # decodable stream, raise an exception.
        if not self._got_a_pad:
            self.read_exc = NoStreamError()
            self.ready_sem.release()  # No effect if we've already started.

    def _new_sample(self, sink):
        """The callback for appsink's "new-sample" signal.
        """
        if self.running:
            # New data is available from the pipeline! Dump it into our
            # queue (or possibly block if we're full).
            buf = sink.emit('pull-sample').get_buffer()

            # We can't use Gst.Buffer.extract() to read the data as it crashes
            # when called through PyGObject. We also can't use
            # Gst.Buffer.extract_dup() because we have no way in Python to free
            # the memory that it returns. Instead we get access to the actual
            # data via Gst.Memory.map().
            mem = buf.get_all_memory()
            success, info = mem.map(Gst.MapFlags.READ)
            if success:
                if isinstance(info.data, memoryview):
                    # We need to copy the data as the memoryview is released
                    # when we call mem.unmap()
                    data = bytes(info.data)
                else:
                    # GStreamer Python bindings <= 1.16 return a copy of the
                    # data as bytes()
                    data = info.data
                mem.unmap(info)
                self.queue.put(data)
            else:
                raise GStreamerError("Unable to map buffer memory while reading the file.")
        return Gst.FlowReturn.OK

    def _unkown_type(self, uridecodebin, decodebin, caps):
        """The callback for decodebin's "unknown-type" signal.
        """
        # This is called *before* the stream becomes ready when the
        # file can't be read.
        streaminfo = caps.to_string()
        if not streaminfo.startswith('audio/'):
            # Ignore non-audio (e.g., video) decode errors.
            return
        self.read_exc = UnknownTypeError(streaminfo)
        self.ready_sem.release()

    def _message(self, bus, message):
        """The callback for GstBus's "message" signal (for two kinds of
        messages).
        """
        if not self.finished:
            if message.type == Gst.MessageType.EOS:
                # The file is done. Tell the consumer thread.
                self.queue.put(SENTINEL)
                if not self.got_caps:
                    # If the stream ends before _notify_caps was called, this
                    # is an invalid file.
                    self.read_exc = NoStreamError()
                    self.ready_sem.release()

            elif message.type == Gst.MessageType.ERROR:
                gerror, debug = message.parse_error()
                if 'not-linked' in debug:
                    self.read_exc = NoStreamError()
                elif 'No such file' in debug:
                    self.read_exc = IOError('resource not found')
                else:
                    self.read_exc = FileReadError(debug)
                self.ready_sem.release()

    # Iteration.

    def __next__(self):
        # Wait for data from the Gstreamer callbacks.
        val = self.queue.get()
        if val == SENTINEL:
            # End of stream.
            raise StopIteration
        return val

    def __iter__(self):
        return self

    # Cleanup.
    def close(self, force=False):
        """Close the file and clean up associated resources.

        Calling `close()` a second time has no effect.
        """
        if self.running or force:
            self.running = False
            self.finished = True

            # Unregister for signals, which we registered for above with
            # `add_signal_watch`. (Without this, GStreamer leaks file
            # descriptors.)
            self.pipeline.get_bus().remove_signal_watch()

            # Stop reading the file.
            self.dec.set_property("uri", None)
            # Block spurious signals.
            self.sink.get_static_pad("sink").disconnect(self.caps_handler)

            # Make space in the output queue to let the decoder thread
            # finish. (Otherwise, the thread blocks on its enqueue and
            # the interpreter hangs.)
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            # Halt the pipeline (closing file).
            self.pipeline.set_state(Gst.State.NULL)

    def __del__(self):
        self.close()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Smoke test.
if __name__ == '__main__':
    for path in sys.argv[1:]:
        path = os.path.abspath(os.path.expanduser(path))
        with GstAudioFile(path) as f:
            print(f.channels)
            print(f.samplerate)
            print(f.duration)
            for s in f:
                print(len(s), ord(s[0]))
