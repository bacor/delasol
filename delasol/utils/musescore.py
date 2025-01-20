# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Author: Bas Cornelissen
# Copyright © 2025 Bas Cornelissen
# -------------------------------------------------------------------
import os
import subprocess
import music21


def run_mscore(*args, mscore_path: str = None) -> subprocess.CompletedProcess:
    """Run the MuseScore executable with the given arguments.

    This function attempts to locate the MuseScore executable either through
    the provided `mscore_path` argument, the `MSCORE_PATH` environment variable,
    or the user settings in the music21 library. If the executable is not found,
    appropriate exceptions are raised.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the MuseScore executable.

    mscore_path : str, optional
        The path to the MuseScore executable. If not provided, the function
        will attempt to determine the path from the environment variable
        `MSCORE_PATH` or the music21 user settings. The path typically looks like
        '/Applications/MuseScore 4.app/Contents/MacOS/mscore'

    Raises
    ------
    FileNotFoundError
        If the MuseScore executable is not found at the specified path or
        if music21 is not configured to use MuseScore.

    ValueError
        If no MuseScore executable is specified and it cannot be found
        through the environment or user settings.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run, containing information about the
        execution of the MuseScore command.
    """
    if mscore_path is None and "MSCORE_PATH" in os.environ:
        mscore_path = os.environ["MSCORE_PATH"]
    elif mscore_path is None:
        us = music21.environment.UserSettings()
        if "musicxmlPath" in us.keys():
            if "MuseScore" in str(us["musicxmlPath"]):
                mscore_path = us["musicxmlPath"]
            else:
                raise FileNotFoundError(
                    f"MuseScore executable not found. Music21 does not seem to use MuseScore, but instead uses: {us['musicxmlPath']}"
                )
    if mscore_path is None:
        raise ValueError(
            "No MuseScore executable was specified. Please specify the mscore_path argument, set the MSCORE_PATH environment variable or configure music21 to use MuseScore"
        )
    if not os.path.exists(mscore_path):
        raise FileNotFoundError(
            f"MuseScore executable does not exist found: {mscore_path}"
        )

    return subprocess.run([mscore_path, *args])
