# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from typing import TypeVar

_T = TypeVar("_T")


def load_pickle_file(fp) -> _T:
    """Centralize file-based pickle deserialization during migration work."""
    return pickle.Unpickler(fp).load()
