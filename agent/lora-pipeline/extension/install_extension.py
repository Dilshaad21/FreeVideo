"""Install the LoRA agent integration without replacing the host route module."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


INTEGRATION_MARKERS = (
    "contracts.update(training.contracts())",
    "value['training'] = dict(base=PREFIX + '/training', guide=PREFIX + '/training/guide')",
    "r['status'] in ('waiting', 'running', 'cancelling')",
    'server._freevideo_training = training.register(routes, guarded, reply, body, port, jobs)',
)
PARTIAL_MARKERS = (
    'from . import training',
    'training.contracts()',
    "value['training']",
    "'training_busy'",
    'training.register(',
    'server._freevideo_training',
)

# These anchors identify the two supported FreeVideo route layouts. Each is
# checked in full before any file is backed up or changed.
ROUTE_ANCHORS = {
    'contracts': '    contracts = contract.contracts()\n',
    'info': '        return reply(await asyncio.to_thread(capabilities, port, jobs))',
    'submit': '    async def submit(request):\n        value = await body(request)',
    'registered': '    server._freevideo_agent_registered = True',
}
BASE_SIGNATURES = (
    '    contracts = contract.contracts()',
    "    @routes.post(PREFIX + '/jobs')",
    '    server._freevideo_agent_registered = True',
)

REPLACEMENTS = {
    'contracts': '    from . import training\n    contracts = contract.contracts()\n    contracts.update(training.contracts())\n',
    'info': "        value = await asyncio.to_thread(capabilities, port, jobs)\n        value['training'] = dict(base=PREFIX + '/training', guide=PREFIX + '/training/guide')\n        return reply(value)",
    'submit': "    async def submit(request):\n        if any(r['status'] in ('waiting', 'running', 'cancelling') for r in server._freevideo_training.runs()):\n            return reply({'error': {'code': 'training_busy', 'message': 'A LoRA is training. Wait or cancel it before generating.'}}, 409)\n        value = await body(request)",
    'registered': '    server._freevideo_training = training.register(routes, guarded, reply, body, port, jobs)\n    server._freevideo_agent_registered = True',
}


class CompatibilityError(RuntimeError):
    """The host agent is unknown or has a partial LoRA integration."""


def _fingerprint(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _validate_payload(payload):
    payload = Path(payload)
    files = {}
    for name in ('training.py', 'training_worker.py'):
        path = payload / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing extension module: {path}')
        source = path.read_text(encoding='utf-8')
        compile(source, str(path), 'exec')
        files[name] = source
    return files


def _prepare_routes(text):
    """Return integrated source and status after validating every expected anchor."""
    counts = [text.count(marker) for marker in INTEGRATION_MARKERS]
    if all(count == 1 for count in counts):
        if any(text.count(signature) != 1 for signature in BASE_SIGNATURES):
            raise CompatibilityError('LoRA markers are present in an unsupported FreeVideo routes.py version.')
        compile(text, 'routes.py', 'exec')
        return text, True
    partial = [marker for marker in PARTIAL_MARKERS if marker in text]
    if partial:
        raise CompatibilityError('Partial LoRA integration detected; refusing to layer another patch. Present markers: '
                                 + ', '.join(repr(marker) for marker in partial))

    missing = [name for name, anchor in ROUTE_ANCHORS.items() if text.count(anchor) != 1]
    if missing:
        raise CompatibilityError('Unsupported FreeVideo routes.py layout; expected each known anchor exactly once '
                                 f'(failed: {", ".join(missing)}).')

    updated = text
    # All anchors were checked before replacements begin.
    for name, anchor in ROUTE_ANCHORS.items():
        updated = updated.replace(anchor, REPLACEMENTS[name], 1)
    compile(updated, 'routes.py', 'exec')
    if any(updated.count(marker) != 1 for marker in INTEGRATION_MARKERS):
        raise CompatibilityError('The supported route anchors did not produce a complete integration.')
    return updated, False


def _status(source, payload):
    routes = Path(source) / 'freevideo_engine' / 'agent' / 'routes.py'
    route_text = routes.read_text(encoding='utf-8')
    fingerprint = _fingerprint(route_text)
    try:
        _validate_payload(payload)
        prepared, installed = _prepare_routes(route_text)
        compatible = True
        reason = None
    except (CompatibilityError, FileNotFoundError, SyntaxError) as error:
        prepared, installed = route_text, False
        compatible = False
        reason = str(error)
    module_paths = [routes.parent / name for name in ('training.py', 'training_worker.py')]
    modules_present = all(path.is_file() for path in module_paths)
    return {
        'installed': bool(installed and modules_present),
        'compatible': compatible,
        'source_fingerprint': fingerprint,
        'reason': reason,
    }, routes, route_text, prepared


def check(source, payload):
    """Return read-only install state suitable for the CLI's JSON output."""
    return _status(source, payload)[0]


def _atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _backup(target, backup):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')
    folder = Path(backup) / stamp
    suffix = 0
    while folder.exists():
        suffix += 1
        folder = Path(backup) / f'{stamp}-{suffix}'
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('routes.py', 'training.py', 'training_worker.py'):
        path = target / name
        if path.is_file():
            shutil.copy2(path, folder / name)
    return folder


def install(source, payload, backup):
    target = Path(source) / 'freevideo_engine' / 'agent'
    status, routes, before, after = _status(source, payload)
    if not status['compatible']:
        raise CompatibilityError(status['reason'])
    modules = _validate_payload(payload)
    folder = _backup(target, backup)
    replaced = []
    try:
        for name, content in modules.items():
            _atomic_write(target / name, content)
            replaced.append(name)
        if after != before:
            _atomic_write(routes, after)
            replaced.append('routes.py')
    except Exception:
        # A multi-file installation can fail between atomic replacements. Put
        # every touched file back from the backup, or remove newly created ones.
        for name in reversed(replaced):
            saved = folder / name
            destination = target / name
            if saved.is_file():
                _atomic_write(destination, saved.read_text(encoding='utf-8'))
            else:
                destination.unlink(missing_ok=True)
        raise
    return {
        'installed': True,
        'compatible': True,
        'source_fingerprint': _fingerprint(after),
        'backup': str(folder),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--check', action='store_true', help='Report compatibility without writing files')
    args = parser.parse_args(argv)
    payload = Path(__file__).parent
    if args.check:
        print(json.dumps(check(args.source, payload), sort_keys=True))
    else:
        print(json.dumps(install(args.source, payload, Path(args.root) / 'agent' / 'training' / 'backups'),
                         sort_keys=True))


if __name__ == '__main__':
    main()
