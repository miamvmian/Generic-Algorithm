from pathlib import Path
from typing import TypeVar, Iterable, Tuple, Union, List


PathLike = TypeVar("PathLike", str, Path)
ListLike = TypeVar('ListLike', list, tuple, Iterable)
ListOrTuple = Union[List[int|float], Tuple[int|float, ...]]
ListOrTuple2 = List[ListOrTuple] | Tuple[ListOrTuple, ...]

# path regulator
def path_regulator(path: PathLike = Path.cwd()) -> PathLike:
    work_path = Path.cwd()
    wp_parents = work_path.parents
    if not path.exists():
        for parent in wp_parents:
            child = work_path.relative_to(parent)
            if path.is_relative_to(child):
                path = parent/path
                break
    return path


# Configuration parameters
folder = Path("ML/curve_fitting_GA")
path = path_regulator(folder)

MAX_WORKERS = 32



