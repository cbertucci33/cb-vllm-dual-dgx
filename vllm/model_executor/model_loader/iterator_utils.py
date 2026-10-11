# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import sys
from contextlib import contextmanager


@contextmanager
def closing_weights_iterator(iterator):
    """Close a weight source without replacing a consumer failure."""
    try:
        yield iterator
    finally:
        error = sys.exc_info()[1]
        close = getattr(iterator, "close", None)
        if close is not None:
            try:
                close()
            except BaseException as cleanup_error:
                if error is None or isinstance(error, GeneratorExit):
                    raise
                note = f"Weight iterator cleanup failed: {cleanup_error!r}"
                add_note = getattr(error, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:
                    logging.getLogger(__name__).warning(note)
