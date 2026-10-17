"""Authenticated FreeVideo endpoints for the isolated replacement POC."""
import asyncio
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time

from .training import read_json,write_json,identity
from .replacement_support import alive,atomic_json,file_lock,mutate,execution_signature,terminate_tree

PREFIX='/freevideo/agent/v1'
ACTIVE=('waiting','running','cancelling','orphaned')


from .replacement_support import validate_request as validate


def contracts():
    from .schema import obj,string,number,array
    empty=obj()
    body=obj(dict(ref=string(100),source=string(1024),character=string(1024),
        box=array(number(0,1),4,4),prompt=string(20000)),('ref','source','character','box'))
    path=obj(dict(id=string(13,pattern='^r[0-9a-f]{12}$')),('id',))
    result={}
    for method,suffix in [('GET','/replacement'),('POST','/replacement/jobs'),('GET','/replacement/jobs/{id}'),
                          ('POST','/replacement/jobs/{id}/cancel'),('GET','/replacement/jobs/{id}/files/{name}'),
                          ('GET','/replacement/jobs/{id}/preview'),('POST','/replacement/jobs/{id}/resume')]:
        fields=dict(body=body if suffix=='/replacement/jobs' else empty,query=empty,auth=True,local=True,path=None)
        if '{id}' in suffix:fields['path']=path
        if '{name}' in suffix:fields['path']=obj(dict(id=path['properties']['id'],name=dict(type='string',enum=['preview.mp4','master.mkv','verification.json'])),('id','name'))
        if method=='GET':fields['body']=None
        if suffix=='/replacement/jobs/{id}':fields['query']=obj(dict(wait=number(0,25,default=0)))
        if suffix.endswith('/preview'):fields['auth']=False
        result[method,suffix]=fields
    return result


class Replacement:
    def __init__(self,root,port):
        self.root=Path(root);self.home=self.root/'agent/replacement';self.port=port
        self.home.mkdir(parents=True,exist_ok=True);self.lock=threading.RLock()

    def status(self):
        setup=read_json(self.home/'setup.json') if (self.home/'setup.json').exists() else {}
        config=read_json(self.home/'backend.json') if (self.home/'backend.json').exists() else {}
        missing=[key for key in ('python','wangp','sam2','sam_checkpoint') if not config.get(key) or not Path(config[key]).exists()]
        for name in ('wan2.2_animate_14B_quanto_bf16_int8.safetensors','models_t5_umt5-xxl-enc-quanto_int8.safetensors'):
            path=self.home/'WanGP/ckpts'/name
            if not path.is_file() or not path.stat().st_size:missing.append(name)
        return dict(api_version='0.2',experimental=True,backend='Wan2.2-Animate via WanGP + SAM 2',
            ready=not missing and setup.get('status')=='succeeded' and setup.get('models',False) and setup.get('runtime_imports',False),
            setup=setup,missing=missing,prefetch=read_json(self.home/'prefetch.json') if (self.home/'prefetch.json').exists() else None,
            finalizer=read_json(self.home/'finish.json') if (self.home/'finish.json').exists() else None,
            recovery='Explicit resume verifies completed stage hashes; interrupted generation is not automatically retried',
            requires=['source video','character reference image','target-person bounding box'],
            limits=dict(seconds=5,shots=1,fps=60),
            quality_validated=False,preservation='Lossless master RGB unchanged outside saved edit-region masks')

    def job(self,name):
        folder=self.home/'runs'/identity(name,'r')
        path=folder/'job.json';row=read_json(path)
        if row['status'] in ACTIVE and time.time()-row.get('heartbeat_at',row['created_at'])>30:
            def reconcile(current):
                if current['status'] not in ACTIVE:return
                if alive(current.get('pid'),current.get('process_created')):return
                if not current.get('pid') and alive(current.get('launch_pid'),current.get('launch_created')):return
                orphan=alive(current.get('child_pid'),current.get('child_created'))
                current.update(status='orphaned' if orphan else 'interrupted',
                    error='Worker exited; cancel the orphaned process before resuming' if orphan else 'Worker exited; explicit resume is available')
            row=mutate(path,reconcile)
        return row

    def launch(self,folder):
        import os,psutil
        env=dict(os.environ)
        # The server imports source dynamically; child interpreters need an explicit path.
        env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
        for key in list(env):
            if key.startswith(('FREEVIDEO_RUNTIME_LOCK_','FREEVIDEO_BOOTSTRAP_')):env.pop(key,None)
        try:
            with (folder/'worker.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-m','freevideo_engine.agent.replacement_worker',
                    '--root',str(self.root),'--job',folder.name],env=env,cwd=str(Path(__file__).resolve().parents[2]),stdout=log,stderr=subprocess.STDOUT)
            try:created=psutil.Process(child.pid).create_time()
            except psutil.NoSuchProcess:created=None
            return mutate(folder/'job.json',lambda row:row.update(launch_pid=child.pid,launch_created=created))
        except Exception as error:
            mutate(folder/'job.json',lambda row:row.update(status='failed',error_code='launch_failed',error=str(error)))
            raise ValueError('Replacement worker could not start; the failed job was preserved') from error

    def start(self,value):
        value=validate(value)
        with self.lock,file_lock(self.home/'submission.lock'):
            # Hash actual input bytes; a reused filename must not silently reuse an old job.
            def source(name):
                path=self.port.input_path(name)
                if not path.is_file() or path.stat().st_size>500*1024**2:raise ValueError('Missing input or file exceeds 500 MB')
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
                return str(path),digest.hexdigest()
            video,video_hash=source(value['source']);character,character_hash=source(value['character'])
            semantic={key:item for key,item in value.items() if key not in ('source','character')}
            fingerprint=hashlib.sha256(json.dumps(dict(semantic,video_hash=video_hash,character_hash=character_hash),sort_keys=True).encode()).hexdigest()
            previous_jobs=[self.job(path.parent.name) for path in (self.home/'runs').glob('*/job.json')]
            for previous in previous_jobs:
                if previous['ref']==value['ref']:
                    if previous['fingerprint']!=fingerprint:raise ValueError('ref already identifies different inputs; use a new ref')
                    return previous
            if any(previous['status'] in ACTIVE for previous in previous_jobs):
                raise ValueError('A replacement is already active; wait or cancel it')
            if not self.status()['ready']:raise ValueError('Replacement runtime is not ready; inspect GET /replacement')
            name='r'+secrets.token_hex(6);folder=self.home/'runs'/name
            row=dict(value,id=name,status='waiting',phase='waiting-for-gpu',created_at=time.time(),fingerprint=fingerprint,
                     video_path=video,character_path=character,video_sha256=video_hash,character_sha256=character_hash,
                     execution_signature=execution_signature(read_json(self.home/'backend.json')),attempt=1)
            atomic_json(folder/'job.json',row)
            return self.launch(folder)

    def resume(self,name):
        with self.lock,file_lock(self.home/'submission.lock'):
            row=self.job(name)
            if row['status']=='succeeded' or row['status'] in ('waiting','running','cancelling'):return row
            if row['status']=='orphaned':raise ValueError('Cancel the orphaned process before resuming')
            if not self.status()['ready']:raise ValueError('Replacement runtime is not ready')
            for path in (self.home/'runs').glob('*/job.json'):
                if path.parent.name!=name and self.job(path.parent.name)['status'] in ACTIVE:
                    raise ValueError('Another replacement is active')
            if row.get('execution_signature')!=execution_signature(read_json(self.home/'backend.json')):
                raise ValueError('Pipeline/configuration changed; use a new ref instead of resuming old stages')
            folder=self.home/'runs'/name
            (folder/'cancel').unlink(missing_ok=True)
            def prepare(current):
                current.update(status='waiting',phase='waiting-for-gpu',pid=None,launch_pid=None,
                    child_pid=None,child_created=None,error=None,error_code=None,heartbeat_at=time.time(),
                    attempt=current.get('attempt',1)+1)
            mutate(folder/'job.json',prepare)
            return self.launch(folder)

    def cancel(self,name):
        row=self.job(name)
        if row['status'] in ACTIVE:
            (self.home/'runs'/name/'cancel').touch()
            if row['status']=='orphaned':
                terminate_tree(row.get('child_pid'),row.get('child_created'))
                mutate(self.home/'runs'/name/'job.json',lambda current:current.update(status='cancelled',finished_at=time.time()))
        return self.job(name)


def register(routes,guarded,reply,body,port,jobs):
    from ..comfy_bridge import installation_root
    service=Replacement(installation_root(),port)
    async def handler(request):
        name=request.match_info.get('id')
        suffix=request.match_info.route.resource.canonical
        try:
            if suffix.endswith('/replacement'):result=await asyncio.to_thread(service.status)
            elif suffix.endswith('/replacement/jobs'):
                value=await body(request)
                result=await asyncio.to_thread(service.start,value)
            elif suffix.endswith('/cancel'):result=await asyncio.to_thread(service.cancel,name)
            elif suffix.endswith('/resume'):
                result=await asyncio.to_thread(service.resume,name)
            elif '/files/' in suffix or suffix.endswith('/preview'):
                from aiohttp import web
                row=await asyncio.to_thread(service.job,name)
                if row['status']!='succeeded' or not row.get('verification',{}).get('verified'):
                    raise ValueError('No verified completed replacement deliverable')
                filename='preview.mp4' if suffix.endswith('/preview') else request.match_info['name']
                if filename not in ('preview.mp4','master.mkv','verification.json'):raise ValueError('Unknown deliverable')
                return web.FileResponse(service.home/'runs'/name/'output'/filename,headers={'Cache-Control':'private, no-store'})
            else:
                deadline=time.monotonic()+min(25,float(request.query.get('wait',0)))
                while True:
                    result=await asyncio.to_thread(service.job,name)
                    if result['status'] not in ACTIVE or time.monotonic()>=deadline:break
                    await asyncio.sleep(.5)
            return reply(result)
        except (ValueError,FileNotFoundError) as error:
            return reply(dict(error=dict(code='replacement_invalid',message=str(error))),400)
    for method,suffix in contracts():routes.route(method,PREFIX+suffix)(guarded(handler))
    return service
