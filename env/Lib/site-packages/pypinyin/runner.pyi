from argparse import ArgumentParser
from typing import Union, Text, ByteString


class NullWriter(object):
    def write(self, string: Union[Text, ByteString]) -> None: ...


def get_parser() -> ArgumentParser: ...


def main() -> None: ...
