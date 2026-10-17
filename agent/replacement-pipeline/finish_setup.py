"""Finish this setup after the resumable core-weight prefetch, without an LLM polling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from replacement_support import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();root=Path(args.root).resolve();home=root/'agent/replacement'
    if os.name!='nt':raise RuntimeError('Windows setup only')
    import msvcrt,psutil
    with (home/'finish.lock').open('a+b') as lock:
        if not lock.tell():lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        state=home/'finish.json'
        def update(**value):
            value.update(updated_at=time.time(),pid=os.getpid())
            atomic_json(state,value)
        deadline=time.monotonic()+4*60*60
        try:
            while True:
                prefetch=home/'prefetch.json'
                row=json.loads(prefetch.read_text()) if prefetch.exists() else {}
                if row.get('status')=='failed':raise RuntimeError('Weight prefetch failed: '+row.get('error','unknown error'))
                probes=[]
                for process in psutil.process_iter(['cmdline']):
                    command=' '.join(process.info['cmdline'] or [])
                    if 'runtime.py' in command and ' probe ' in command:probes.append(process.pid)
                if row.get('status')=='succeeded' and not probes:break
                if time.monotonic()>deadline:raise TimeoutError('Setup wait exceeded four hours; rerun prefetch/finish after checking connectivity')
                update(status='waiting',phase='core-weights' if row.get('status')!='succeeded' else 'existing-probe')
                time.sleep(20)
            update(status='running',phase='final-runtime-and-assets')
            env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1')
            env['PATH']=os.pathsep.join(p for p in env.get('PATH','').split(os.pathsep)
                if '/openai/codex' not in p.replace('\\','/').lower())
            with (home/'final-setup.log').open('ab') as log:
                subprocess.run([sys.executable,str(Path(__file__).parent/'setup_windows.py'),
                    '--root',str(root),'--skip-install','--models'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7500)
            update(status='succeeded',phase='ready')
        except BaseException as error:
            update(status='failed',error=str(error));raise


if __name__=='__main__':main()
