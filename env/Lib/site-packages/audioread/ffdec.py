# This file is part of audioread.
# Copyright 2014, Adrian Sampson.
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

"""Read audio data using the ffmpeg command line tool via its standard
output.
"""

import queue
import re
import subprocess
import sys
import threading
import time
from io import DEFAULT_BUFFER_SIZE

from .exceptions import DecodeError
from .base import AudioFile

COMMANDS = ('ffmpeg', 'avconv')

if sys.platform == "win32":
    PROC_FLAGS = 0x08000000
else:
    PROC_FLAGS = 0


class FFmpegError(DecodeError):
    pass


class CommunicationError(FFmpegError):
    """Raised when the output of FFmpeg is not parseable."""


class UnsupportedError(FFmpegError):
    """The file could not be decoded by FFmpeg."""


class NotInstalledError(FFmpegError):
    """Could not find the ffmpeg binary."""


class ReadTimeoutError(FFmpegError):
    """Reading from the ffmpeg command-line tool timed out."""


class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue.
    """
    def __init__(self, fh, blocksize=1024, discard=False):
        super().__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                # Stream closed (EOF).
                break


def popen_multiple(commands, command_args, *args, **kwargs):
    """Like `subprocess.Popen`, but can try multiple commands in case
    some are not available.

    `commands` is an iterable of command names and `command_args` are
    the rest of the arguments that, when appended to the command name,
    make up the full first argument to `subprocess.Popen`. The
    other positional and keyword arguments are passed through.
    """
    for i, command in enumerate(commands):
        cmd = [command] + command_args
        try:
            return subprocess.Popen(cmd, *args, **kwargs)
        except OSError:
            if i == len(commands) - 1:
                # No more commands to try.
                raise


def available():
    """Detect whether the FFmpeg backend can be used on this system.
    """
    try:
        proc = popen_multiple(
            COMMANDS,
            ['-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=PROC_FLAGS,
        )
    except OSError:
        return False
    else:
        proc.communicate()
        return proc.returncode == 0


# For Windows error switch management, we need a lock to keep the mode
# adjustment atomic.
windows_error_mode_lock = threading.Lock()


class FFmpegAudioFile(AudioFile):
    """An audio file decoded by the ffmpeg command-line utility."""
    def __init__(self, filename, block_size=DEFAULT_BUFFER_SIZE):
        # On Windows, we need to disable the subprocess's crash dialog
        # in case it dies. Passing SEM_NOGPFAULTERRORBOX to SetErrorMode
        # disables this behavior.
        windows = sys.platform.startswith("win")
        if windows:
            windows_error_mode_lock.acquire()
            SEM_NOGPFAULTERRORBOX = 0x0002
            import ctypes
            # We call SetErrorMode in two steps to avoid overriding
            # existing error mode.
            previous_error_mode = \
                ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
            ctypes.windll.kernel32.SetErrorMode(
                previous_error_mode | SEM_NOGPFAULTERRORBOX
            )

        try:
            self.proc = popen_multiple(
                COMMANDS,
                ['-i', filename, '-f', 's16le', '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=PROC_FLAGS,
            )

        except OSError:
            raise NotInstalledError()

        finally:
            # Reset previous error mode on Windows. (We can change this
            # back now because the flag was inherited by the subprocess;
            # we don't need to keep it set in the parent process.)
            if windows:
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetErrorMode(previous_error_mode)
                finally:
                    windows_error_mode_lock.release()

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        self._get_info()

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ReadTimeoutError('ffmpeg output: {}'.format(
                            b''.join(self.stderr_reader.queue.queue)
                        ))
                    else:
                        start_time = end_time
                        # Keep waiting.
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise CommunicationError("stream info not found")

            # In Python 3, result of reading from stderr is bytes.
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')

            line = line.strip().lower()

            if 'no such file' in line:
                raise OSError('file not found')
            elif 'invalid data found' in line:
                raise UnsupportedError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        # Sample rate.
        match = re.search(r'(\d+) hz', s)
        if match:
            self.samplerate = int(match.group(1))
        else:
            self.samplerate = 0

        # Channel count.
        match = re.search(r'hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels = 2
            else:
                cmatch = re.match(r'(\d+)\.?(\d)?', mode)
                if cmatch:
                    self.channels = sum(map(int, cmatch.group().split('.')))
                else:
                    self.channels = 1
        else:
            self.channels = 0

        # Duration.
        match = re.search(
            r'duration: (\d+):(\d+):(\d+).(\d)', s
        )
        if match:
            durparts = list(map(int, match.groups()))
            duration = (
                durparts[0] * 60 * 60 +
                durparts[1] * 60 +
                durparts[2] +
                float(durparts[3]) / 10
            )
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        if hasattr(self, 'proc'):
            # First check the process's execution status before attempting to
            # kill it. This fixes an issue on Windows Subsystem for Linux where
            # ffmpeg closes normally on its own, but never updates
            # `returncode`.
            self.proc.poll()

            # Kill the process if it is still running.
            if self.proc.returncode is None:
                self.proc.kill()
                self.proc.wait()

            # Wait for the stream-reading threads to exit. (They need to
            # stop reading before we can close the streams.)
            if hasattr(self, 'stderr_reader'):
                self.stderr_reader.join()
            if hasattr(self, 'stdout_reader'):
                self.stdout_reader.join()

            # Close the stdout and stderr streams that were opened by Popen,
            # which should occur regardless of if the process terminated
            # cleanly.
            self.proc.stdout.close()
            self.proc.stderr.close()

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_data()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
