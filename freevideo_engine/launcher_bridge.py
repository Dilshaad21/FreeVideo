"""File handoff between the launcher and the ComfyUI server it started.

The server only records an explicit browser request; the launcher itself
checks, downloads the verified release and restarts. Standard library only.
"""
import json
import os
from pathlib import Path
import secrets
import shutil
import time

from .monitoring import save
from .runtime_alias import is_junction, checked_real_path, AliasUnavailable
from .long_paths import io_path, plain, resolved

ENV = 'FREEVIDEO_LAUNCHER_BRIDGE'
# The launcher rewrites its status at least this often while it runs.
HEARTBEAT_SECONDS = 5
FRESH_SECONDS = 20


def create(root):
    selected = Path(root)
    root = checked_real_path(root)
    if root != selected:
        root = resolved(root)
    parent = io_path(root) / 'bridge'
    if is_junction(parent):
        raise ValueError('Launcher bridge folder is redirected; retained.')
    parent.mkdir(parents=True, exist_ok=True)
    # Folders of launchers that ended without cleanup are only advisory.
    for old in parent.iterdir():
        try:
            if not is_junction(old) and old.is_dir() and time.time() - old.stat().st_mtime > 86400:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    path = parent / ('%d-%s' % (os.getpid(), secrets.token_hex(4)))
    path.mkdir()
    return Path(plain(path))


def remove(path):
    try:
        path = io_path(checked_real_path(path)) if path else path
    except AliasUnavailable:
        return
    if path and not is_junction(path) and not is_junction(Path(path).parent):
        shutil.rmtree(path, ignore_errors=True)


def write_status(path, value):
    save(io_path(checked_real_path(path)) / 'status.json', dict(value, updated_at=time.time()))


def read_status(path, *, max_age=FRESH_SECONDS):
    """A recent launcher status, or None when no launcher is listening."""
    try:
        value = json.loads((io_path(checked_real_path(path)) / 'status.json').read_text(encoding='utf-8'))
        if isinstance(value, dict) and abs(time.time() - float(value.get('updated_at', 0))) <= max_age:
            return value
    except (OSError, ValueError, TypeError):
        pass
    return None


def request(path, action='update'):
    if action not in ('update', 'cancel'):
        raise ValueError('Unknown launcher request')
    save(io_path(checked_real_path(path)) / 'request.json', dict(action=action, at=time.time()))


def take_request(path):
    try:
        file = io_path(checked_real_path(path)) / 'request.json'
    except AliasUnavailable:
        return None
    try:
        value = json.loads(file.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        value = None
    try:
        file.unlink()
    except OSError:
        pass
    return value if isinstance(value, dict) and value.get('action') in ('update', 'cancel') else None
