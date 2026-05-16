"""
General utility functions for the computer-vision pipeline.

These functions are intentionally simple and reusable across notebooks.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional


def ensure_dir(path: str | Path) -> Path:
    """
    Create a folder if it does not exist.

    Parameters
    ----------
    path : str or Path
        Folder path to create.

    Returns
    -------
    Path
        Created or existing folder path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Parameters
    ----------
    path : str or Path
        JSON file path.

    Returns
    -------
    dict
        JSON content as a dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Data to save.
    path : str or Path
        Output JSON file path.
    indent : int
        JSON indentation level.

    Returns
    -------
    Path
        Saved file path.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)

    return path


def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    """
    Load the project configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the local configuration file.

    Returns
    -------
    dict
        Project configuration.
    """
    return load_json(path)


def get_path(config: Dict[str, Any], section: str, key: str) -> Path:
    """
    Get a path from the configuration dictionary.

    Example
    -------
    get_path(config, "paths", "raw_data_dir")

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    section : str
        Top-level section name.
    key : str
        Key inside the selected section.

    Returns
    -------
    Path
        Path object.
    """
    try:
        return Path(config[section][key])
    except KeyError as error:
        raise KeyError(f"Missing config value: {section}.{key}") from error


def safe_stem(path: str | Path) -> str:
    """
    Return the file name without extension.

    Parameters
    ----------
    path : str or Path
        Input file path.

    Returns
    -------
    str
        File stem.
    """
    return Path(path).stem


def list_files(
    folder: str | Path,
    pattern: str = "*",
    recursive: bool = False
) -> list[Path]:
    """
    List files in a folder.

    Parameters
    ----------
    folder : str or Path
        Folder to scan.
    pattern : str
        File pattern, for example '*.png'.
    recursive : bool
        Whether to search recursively.

    Returns
    -------
    list[Path]
        Sorted list of matching files.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return sorted([file for file in files if file.is_file()])


def first_existing_path(paths: list[str | Path]) -> Optional[Path]:
    """
    Return the first path that exists.

    Parameters
    ----------
    paths : list
        Candidate paths.

    Returns
    -------
    Path or None
        First existing path, or None if no path exists.
    """
    for path in paths:
        path = Path(path)
        if path.exists():
            return path

    return None
