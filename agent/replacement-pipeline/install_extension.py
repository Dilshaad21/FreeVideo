"""Install replacement endpoints into an idle, known FreeVideo source layout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid
from replacement_support import file_lock,digest,read_json


def prepare(text):
    patches=[
        ('    contracts.update(training.contracts())','    contracts.update(training.contracts())\n    from . import replacement\n    contracts.update(replacement.contracts())'),
        ('    server._freevideo_training = training.register(routes, guarded, reply, body, port, jobs)',
         '    server._freevideo_training = training.register(routes, guarded, reply, body, port, jobs)\n    server._freevideo_replacement = replacement.register(routes, guarded, reply, body, port, jobs)')]
    for before,after in patches:
        if after in text:continue
        if text.count(before)!=1:raise RuntimeError('Unknown FreeVideo integration layout')
        text=text.replace(before,after)
    ast.parse(text)
    return text


def verify_payload(payload):
    expected=read_json(payload/'source-sha256.json')
    for name,checksum in expected.items():
        path=payload/name
        if not path.resolve().is_relative_to(payload.resolve()) or path.is_symlink() or not path.is_file() or digest(path)!=checksum:
            raise RuntimeError('Package integrity check failed: '+name)
    required={'replacement.py','replacement_worker.py','replacement_support.py','runtime.py','composite.py'}
    if not required.issubset(expected):raise RuntimeError('Incomplete package manifest')


def publish(path,data):
    temporary=path.with_suffix('.replacement.tmp')
    temporary.write_bytes(data);temporary.replace(path)


def install(root,source,payload):
    verify_payload(payload)
    with file_lock(root/'agent/replacement/submission.lock',timeout=0):
        return _install(root,source,payload)


def _install(root,source,payload):
    # Read-only readiness check; never stop jobs to install this POC.
    import urllib.request
    token=(root/'agent/token').read_text(encoding='utf-8-sig').strip()
    def api(path):
        req=urllib.request.Request('http://127.0.0.1:8188/freevideo/agent/v1'+path,headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)
    queue=api('/queue')
    if any(queue.get(key) for key in ('running','pending','agent_jobs','vlm_busy')):raise RuntimeError('FreeVideo is busy')
    if api('/training').get('running'):raise RuntimeError('A trainer is active')
    try:
        status=api('/replacement')
        import psutil
        for p in psutil.process_iter(['cmdline']):
            if 'freevideo_engine.agent.replacement_worker' in ' '.join(p.info['cmdline'] or []):raise RuntimeError('Replacement worker active')
    except urllib.error.HTTPError as error:
        if error.code!=404:raise
    folder=source/'freevideo_engine/agent';routes=folder/'routes.py'
    before=routes.read_text(encoding='utf-8');after=prepare(before)
    modules=['replacement.py','replacement_worker.py','replacement_support.py']
    for name in modules:ast.parse((payload/name).read_text())
    backup=root/'agent/replacement/backups'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    backup.mkdir(parents=True)
    targets=[routes]+[folder/name for name in modules]
    for path in targets:
        if path.exists():shutil.copy2(path,backup/path.name)
    changed=[]
    try:
        for path,data in [(routes,after.encode('utf-8'))]+[(folder/name,(payload/name).read_bytes()) for name in modules]:
            publish(path,data);changed.append(path)
    except BaseException:
        for path in reversed(changed):
            old=backup/path.name
            if old.exists():shutil.copy2(old,path)
            else:path.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(installed=True,restart_required=True,backup=str(backup),routes_sha256=hashlib.sha256(after.encode()).hexdigest())))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--source',required=True)
    args=parser.parse_args();install(Path(args.root),Path(args.source),Path(__file__).resolve().parent)
