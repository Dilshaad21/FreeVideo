"""Publish closed files without restarting downloads on Windows sharing locks."""
from pathlib import Path
import time
from .system import windows
from .long_paths import io_path, plain


def publish(source, destination):
    source, destination = io_path(source), io_path(destination)
    for attempt in range(8):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('Destination already exists; both files retained: ' + plain(destination))
        try:
            source.rename(destination)
            return
        except OSError as error:
            if not windows() or getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            # Defender/indexers can briefly open a just-closed model. Retry only
            # publication, never the transfer. Permanent permission errors stop.
            time.sleep(.1 * (attempt + 1))
