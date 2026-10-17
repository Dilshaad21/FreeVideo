"""Check/reapply the extension on Windows; restart only when explicitly requested and idle."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import urllib.error
import time


def api(root, path, timeout=15):
    token = (root / 'agent/token').read_text(encoding='utf-8-sig').strip()
    request = urllib.request.Request('http://127.0.0.1:8188/freevideo/agent/v1' + path,
                                     headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--source', required=True, help='Exact active FreeVideo source directory')
    p.add_argument('--check', action='store_true')
    p.add_argument('--restart', action='store_true')
    args = p.parse_args()
    root, source = Path(args.root), Path(args.source)
    extension = Path(__file__).resolve().parent / 'extension'
    command = [sys.executable, str(extension / 'install_extension.py'), '--source', str(source), '--root', str(root)]
    if args.check:
        return subprocess.call(command + ['--check'])
    if os.name != 'nt':
        raise RuntimeError('Run this installer on the Windows FreeVideo host')
    # Refuse to interrupt GPU work or modify code used by an active trainer.
    try:
        queue = api(root, '/queue', timeout=3)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ConnectionRefusedError) and getattr(error.reason, 'winerror', None) != 10061:
            raise
        queue = {}
    if queue.get('running') or queue.get('pending') or queue.get('agent_jobs') or queue.get('vlm_busy'):
        raise RuntimeError('FreeVideo is busy; install again after current work finishes')
    import psutil
    for process in psutil.process_iter(['cmdline']):
        commandline = ' '.join(process.info.get('cmdline') or [])
        if 'freevideo_engine.agent.training_worker' in commandline:
            raise RuntimeError('A trainer is running; wait before installing')
    subprocess.run(command, check=True)
    if not args.restart:
        print(json.dumps({'installed': True, 'restart_required': True}))
        return 0
    launch = json.loads((Path.home() / 'Documents/FreeVideo/launcher.json').read_text(encoding='utf-8-sig'))
    comfy = Path(launch['comfy'])
    # Only stop the listener on the configured FreeVideo port, after the idle checks.
    subprocess.run(['powershell', '-NoProfile', '-Command',
        '$listeners=@(Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue); foreach($owner in ($listeners | Select-Object -ExpandProperty OwningProcess -Unique)){Stop-Process -Id $owner -Force -ErrorAction Stop}; exit 0'], check=True)
    # WMI starts the server outside the SSH session's Windows job object.
    # DETACHED_PROCESS alone is insufficient: sshd can still terminate it on logout.
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    log_path = root / 'agent/training/pipeline-server.log'
    child = r"$env:PATH=($env:PATH -split ';' | Where-Object {$_ -notmatch 'OpenAI\\Codex'}) -join ';'; "
    child += "$env:FREEVIDEO_HOME=" + quote(root) + "; $env:PYTHONPATH=" + quote(source) + "; "
    child += "$env:PYTHONUTF8='1'; $env:PYTHONNOUSERSITE='1'; Set-Location " + quote(comfy) + "; & "
    child += quote(launch['python']) + " -u -B " + quote(comfy / 'main.py')
    child += " --listen 0.0.0.0 --port 8188 --disable-auto-launch --disable-all-custom-nodes --whitelist-custom-nodes FreeVideo *> " + quote(log_path)
    encoded = base64.b64encode(child.encode('utf-16le')).decode()
    outer = "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="
    outer += quote('powershell.exe -NoProfile -EncodedCommand ' + encoded) + "; CurrentDirectory=" + quote(comfy) + "} | ConvertTo-Json -Compress"
    spawned = subprocess.run(['powershell', '-NoProfile', '-EncodedCommand',
                              base64.b64encode(outer.encode('utf-16le')).decode()], capture_output=True, check=True)
    if json.loads(spawned.stdout.decode('utf-8-sig')).get('ReturnValue') != 0:
        raise RuntimeError('Could not launch the standalone server process')
    for _ in range(30):
        try:
            result = api(root, '/training', timeout=1)
            if result.get('api_version') == '1.1':
                print(json.dumps({'installed': True, 'api_version': '1.1', 'ready': result.get('ready')}))
                return 0
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError('Server restart not verified; inspect agent/training/pipeline-server.log')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'error': str(error)}))
        sys.exit(1)
