"""Declare widely-used parsers"""
from typing import Callable, List, TypeVar


T = TypeVar("T")


def parse_list(file_path: str, typecast_func: Callable[[str], T]) -> List[T]:
    """Parse the file into the list

    Parameters
    ----------
    file_path : str
        Path of the file
    typecast_func : Callable[[str], T]
        Type-casting function

    Returns
    -------
    List[T]
        Output list
    """
    info_list = []
    with open(file_path, "r") as f:
        info_list = [typecast_func(line.strip()) for line in f.readlines()]

    return info_list
