"""Compact installation storage; never remove borrowed models or user outputs."""
import json
import os
from pathlib import Path
import stat
import time

from .monitoring import save
from .runtime_alias import is_junction, checked_real_path, AliasUnavailable
from .long_paths import io_path, resolved, plain, plain_text


def conversion_source(row):
    name = row['file']
    return name.endswith('.safetensors') and name.startswith(('h3-base/transformer/', 'stage-dmd-step-250/'))


def fingerprint(path):
    path = io_path(checked_real_path(path))
    info = path.stat()
    result = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                  inode=info.st_ino, device=info.st_dev)
    if os.name == 'nt':
        from .win32 import file_change_time_ns
        result['change_time_ns'] = file_change_time_ns(path)
    return result


def managed_model(root, path):
    """Only regular, unshared files beneath this installation's models directory."""
    try:
        selected = Path(root)
        root, path = checked_real_path(root), checked_real_path(path)
        if root != selected:
            from .installation_cleanup import root_path
            root, path = root_path(root, path)  # Also expand an authenticated 8.3 target prefix.
        else:
            root, path = resolved(root), io_path(path).absolute()
        path.relative_to(root / 'models')
        if path.resolve() != path:
            return False
        current = path
        while current != root:
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                return False
            current = current.parent
        info = path.stat()
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    except (OSError, ValueError):
        return False


def ownership(root):
    selected = Path(root)
    root = checked_real_path(root)
    if root != selected:
        root = resolved(root)
    path = io_path(root) / '.freevideo' / 'downloaded-models.json'
    if not path.exists():
        return Path(plain(path)), {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('root') != plain(resolved(root)) or not isinstance(value.get('files'), dict):
        raise ValueError('Invalid model ownership record; models retained')
    return Path(plain(path)), value['files']


def record_download(root, path, row, *, origin='download'):
    selected = Path(root)
    try:
        root, path = checked_real_path(root), checked_real_path(path)
    except AliasUnavailable:
        return
    if root != selected:
        from .installation_cleanup import root_path
        root, path = root_path(root, path)
    # Called only after a previously absent target was downloaded/copied and verified.
    if not conversion_source(row) or not managed_model(root, path):
        return
    ledger, files = ownership(root)
    files[plain(io_path(path).absolute())] = dict(fingerprint(path), sha256=row.get('sha256'),
                                           git_blob=row.get('git_blob'), source_file=row['file'], origin=origin)
    save(ledger, {'root': plain(resolved(root)), 'files': files})


def compact_sources(plan, rows, report):
    """Caller must first verify the complete FP8 cache and pass GPU setup probes.

    Deletion uses an allowlist of pinned conversion-only filenames and a matching
    download receipt. External directories, preexisting/shared/replaced files,
    decoder/encoder weights, configs, failed attempts and outputs are retained.
    """
    result = {'mode': plan.get('storage', 'compact'), 'deleted': [], 'retained': [],
              'released_bytes': 0, 'started_epoch': time.time()}
    try:
        plan = dict(plan, root=checked_real_path(plan['root']), model_dir=checked_real_path(plan['model_dir']))
        report = checked_real_path(report)
    except AliasUnavailable as error:
        result['retained'].append({'reason': str(error)})
        return result  # Even the report path may be inside the invalid alias.
    from .network import hash_file
    from .installation_cleanup import root_path, local_path
    root, chosen_models = root_path(plan['root'], plan['model_dir'])
    model_dir = resolved(chosen_models)
    internal_redirect = (chosen_models.is_relative_to(root / 'models')
                         and not local_path(root, chosen_models))
    if internal_redirect or is_junction(root / 'models') or not model_dir.is_relative_to(root / 'models'):
        result['retained'].append({'reason': 'Model directory is outside installation models or redirected'})
        save(report, result)
        return result
    if result['mode'] != 'compact':
        save(report, result)
        return result
    try:
        ledger, files = ownership(root)
    except (OSError, ValueError, TypeError) as error:
        result['retained'].append({'reason': plain_text(error)})
        save(report, result)
        return result
    # Record intent before modifying any file; update after each successful unlink.
    save(report, result)
    for row in rows:
        if not conversion_source(row):
            continue
        path = model_dir / row['file']
        if not path.exists():
            continue
        expected = files.get(plain(path), {})
        current = fingerprint(path)
        if (not isinstance(expected, dict) or not managed_model(root, path)
                or 'change_time_ns' in current and current['change_time_ns'] is None
                or expected.get('source_file') != row['file'] or expected.get('bytes') != row['bytes']
                or expected.get('sha256') != row.get('sha256') or expected.get('git_blob') != row.get('git_blob')
                or any(expected.get(k) != v for k, v in current.items())):
            result['retained'].append({'file': plain(path), 'reason': 'Not an unchanged, unshared file downloaded by this installation'})
            continue
        try:
            # Destructive cleanup cannot trust a metadata cache hit. Equal-size
            # edits may share a filesystem timestamp tick with the receipt.
            if row.get('sha256'):
                matches = hash_file(path, discard_cache=True) == row['sha256']
            elif row.get('git_blob'):
                matches = hash_file(path, 'sha1', git_blob=True, discard_cache=True) == row['git_blob']
            else:
                raise OSError('No pinned content hash; source retained')
            if not matches:
                raise OSError('Source content changed; retained without deletion')
            result['pending'] = {'file': plain(path), 'bytes': row['bytes']}
            save(report, result)
            # Hashing can take time for model shards. Recheck both identity and
            # sharing immediately before unlink, after saving deletion intent.
            if fingerprint(path) != current or not managed_model(root, path):
                raise OSError('File changed or became shared during cleanup')
            path.unlink()
        except OSError as error:
            result['retained'].append({'file': plain(path), 'reason': plain_text(error)})
            result['pending'] = None
            save(report, result)
            continue
        result['pending'] = None
        result['deleted'].append({'file': plain(path), 'bytes': row['bytes'], 'source_file': row['file']})
        result['released_bytes'] += row['bytes']
        files.pop(plain(path), None)
        save(report, result)
        save(ledger, {'root': plain(root), 'files': files})
    result['completed_epoch'] = time.time()
    save(report, result)
    return result
