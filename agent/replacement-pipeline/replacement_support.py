"""Durable replacement state, verified stage receipts and process ownership."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

VERSION='0.2.0'


def validate_request(value):
    if not isinstance(value,dict):raise ValueError('Request must be an object')
    if set(value)-{'ref','source','character','box','prompt'}:raise ValueError('Unknown request fields')
    box=value.get('box',[])
    if (not isinstance(box,list) or len(box)!=4 or any(type(x) not in (int,float) or not 0<=x<=1 for x in box)
            or box[0]>=box[2] or box[1]>=box[3]):
        raise ValueError('box must be normalized [left, top, right, bottom] for one person')
    for key,limit in [('ref',100),('source',1024),('character',1024)]:
        if not isinstance(value.get(key),str) or not value[key].strip() or len(value[key])>limit or '\x00' in value[key]:
            raise ValueError(key+' must be nonempty text within its length limit')
    prompt=value.get('prompt','A person performing the original action with natural lighting.')
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>20000 or '\x00' in prompt:
        raise ValueError('prompt must contain 1–20000 characters of text')
    return dict(ref=value['ref'],source=value['source'],character=value['character'],box=[float(x) for x in box],prompt=prompt)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+secrets.token_hex(8)+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        for attempt in range(30):
            try:temporary.replace(path);break
            except PermissionError:
                if attempt==29:raise
                time.sleep(.05)
    finally:temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@contextmanager
def file_lock(path,timeout=30):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            if stream.tell()==0:stream.write(b'0');stream.flush()
        else:import fcntl
        deadline=time.monotonic()+timeout
        while True:
            try:
                if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                else:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except (BlockingIOError,OSError):
                if time.monotonic()>=deadline:raise TimeoutError('Another process holds '+path.name)
                time.sleep(.05)
        try:yield
        finally:
            if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)


def mutate(path,callback):
    with file_lock(Path(path).with_suffix('.lock')):
        row=read_json(path);result=callback(row)
        atomic_json(path,row)
        return row if result is None else result


def alive(pid,created):
    if not pid or created is None:return False
    import psutil
    try:
        process=psutil.Process(pid)
        return process.is_running() and abs(process.create_time()-created)<.1 and process.status()!=psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def terminate_tree(pid,created):
    if not alive(pid,created):return
    import psutil
    try:
        parent=psutil.Process(pid);processes=parent.children(recursive=True)+[parent]
        for process in processes:
            try:process.terminate()
            except psutil.NoSuchProcess:pass
        _,remaining=psutil.wait_procs(processes,timeout=5)
        for process in remaining:
            try:process.kill()
            except psutil.NoSuchProcess:pass
        psutil.wait_procs(remaining,timeout=5)
    except psutil.NoSuchProcess:pass


def bind_worker_tree():
    """Windows kills all descendants if this worker crashes, including GPU children.

    Keep the returned raw handle open until process exit. Closing it early also
    terminates this worker. Nested jobs are supported on the target Windows host.
    """
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_=[('per_process',ctypes.c_longlong),('per_job',ctypes.c_longlong),('flags',w.DWORD),
            ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',w.DWORD),
            ('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(ctypes.Structure):_fields_=[(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read','write','other')]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                  ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];kernel.CreateJobObjectW.restype=w.HANDLE
    kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE]
    kernel.GetCurrentProcess.restype=w.HANDLE
    kernel.CloseHandle.argtypes=[w.HANDLE]
    handle=kernel.CreateJobObjectW(None,None)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    info=Extended();info.basic.flags=0x2000
    if (not kernel.SetInformationJobObject(handle,9,ctypes.byref(info),ctypes.sizeof(info))
            or not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess())):
        error=ctypes.get_last_error();kernel.CloseHandle(handle);raise ctypes.WinError(error)
    return handle


def execution_signature(config):
    source=Path(config['source'])
    files={name:digest(source/name) for name in ('runtime.py','composite.py','replacement_support.py','replacement_worker.py')}
    pins={}
    for key in ('wangp','sam2'):
        path=Path(config[key])/'PINNED_COMMIT'
        pins[key]=path.read_text() if path.exists() else 'unrecorded'
    assets={}
    home=Path(config['home'])
    for path in [Path(config['sam_checkpoint']),
                 home/'WanGP/ckpts/wan2.2_animate_14B_quanto_bf16_int8.safetensors',
                 home/'WanGP/ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors']:
        stat=path.stat();assets[str(path)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    dependencies=home/'dependencies.txt'
    return canonical(dict(version=VERSION,config=config,code=files,pins=pins,assets=assets,
                          dependencies=digest(dependencies) if dependencies.exists() else None))


def manifest(paths):
    files=[]
    for value in paths:
        path=Path(value)
        if path.is_symlink():raise ValueError('Symlinks are not stage artifacts')
        if path.is_dir():
            children=sorted(path.rglob('*'))
            if any(p.is_symlink() for p in children):raise ValueError('Symlink in stage output')
            children=[p for p in children if p.is_file()]
            if not children:raise ValueError('Empty stage output: '+str(path))
        elif path.is_file():children=[path]
        else:raise ValueError('Missing stage output: '+str(path))
        files.extend(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)) for p in children)
    return files


class Journal:
    def __init__(self,folder,signature):
        self.folder=Path(folder)/'journal';self.folder.mkdir(exist_ok=True)
        self.chain=signature;self.receipts={}

    def key(self,action,request):return canonical(dict(parent=self.chain,action=action,request=request))

    def reusable(self,name,key,outputs):
        path=self.folder/(name+'.json')
        if not path.exists():return False
        try:
            receipt=read_json(path)
            valid=receipt.get('key')==key and receipt.get('files')==manifest(outputs)
        except (OSError,ValueError):valid=False
        if valid:self.chain=canonical(receipt);self.receipts[name]=receipt
        return valid

    def complete(self,name,key,outputs):
        receipt=dict(key=key,files=manifest(outputs))
        atomic_json(self.folder/(name+'.json'),receipt);self.chain=canonical(receipt);self.receipts[name]=receipt

    def verify_all(self):
        for receipt in self.receipts.values():
            for file in receipt['files']:
                target=Path(file['path'])
                if target.is_symlink() or not target.is_file() or target.stat().st_size!=file['bytes'] or digest(target)!=file['sha256']:
                    raise ValueError('Stage artifact changed after verification: '+target.name)


class ResourceMonitor:
    def __init__(self,config):
        self.config=config;self.last=0;self.hot_since=None;self.low_since=None;self.summary={}

    def sample(self,force=False):
        import psutil,subprocess
        now=time.monotonic()
        if not force and now-self.last<10:return self.summary
        self.last=now;free=psutil.virtual_memory().available
        self.summary['min_available_ram_bytes']=min(free,self.summary.get('min_available_ram_bytes',free))
        self.low_since=(self.low_since if self.low_since is not None else now) if free<self.config.get('ram_floor_bytes',512*1024**2) else None
        if self.low_since is not None and now-self.low_since>=15:raise RuntimeError('resource_limit: available RAM remained below the configured floor')
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],timeout=4,text=True,stderr=subprocess.DEVNULL)
            temperature,vram,utilization=map(int,raw.splitlines()[0].split(','))
            self.summary.update(temperature_c=temperature,vram_mib=vram,utilization_percent=utilization)
            self.summary['peak_temperature_c']=max(temperature,self.summary.get('peak_temperature_c',temperature))
            self.summary['peak_vram_mib']=max(vram,self.summary.get('peak_vram_mib',vram))
            self.hot_since=(self.hot_since if self.hot_since is not None else now) if temperature>=self.config.get('max_temperature_c',87) else None
            if self.hot_since is not None and now-self.hot_since>=30:raise RuntimeError('resource_limit: GPU stayed above the configured temperature limit')
        except (OSError,subprocess.SubprocessError,ValueError):self.summary['gpu_sensor_available']=False
        else:self.summary['gpu_sensor_available']=True
        return self.summary
