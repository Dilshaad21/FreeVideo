"""Photo datasets and isolated H3 LoRA runs behind the existing agent authentication."""
import asyncio
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
import struct


PREFIX = '/freevideo/agent/v1'
EXTENSION_VERSION = '1.1'


def inspect_adapter(path):
    """Inspect H3 safetensors factors without loading model weights."""
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated adapter header')
        size = struct.unpack('<Q', prefix)[0]
        if size > 16 * 1024**2:
            raise ValueError('Invalid adapter header size')
        header = json.loads(stream.read(size))
    down = [k for k in header if k.startswith('lora_unet_blocks_') and k.endswith('.lora_down.weight')]
    if not down:
        raise ValueError('Not a recognized MiniMax H3 adapter')
    for key in down:
        a = header[key].get('shape', [])
        b = header.get(key.replace('.lora_down.weight', '.lora_up.weight'), {}).get('shape', [])
        if len(a) != 2 or len(b) != 2 or a[0] != b[1] or min(a+b) <= 0:
            raise ValueError('Incomplete or mismatched adapter factors')
    offsets = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
    if not offsets or path.stat().st_size < 8 + size + max(offsets):
        raise ValueError('Truncated adapter tensors')
    return dict(verified=True, bytes=path.stat().st_size, factor_pairs=len(down), family='minimax-h3')
ANGLES = ('front', 'left-three-quarter', 'right-three-quarter', 'left-profile', 'right-profile',
          'slightly-above', 'slightly-below', 'waist-up', 'full-body', 'back', 'other')
GUIDE = {
    'recommended_photos': '20–30 distinct, sharp photos per person; 10–15 can be a first experiment.',
    'angles': {'front': 4, 'left-three-quarter': 3, 'right-three-quarter': 3,
               'left-profile': 2, 'right-profile': 2, 'slightly-above': 1,
               'slightly-below': 1, 'waist-up': 3, 'full-body': 3},
    'notes': [
        'One person or subject per dataset. Use a different trigger word for every subject.',
        'Keep eyes and facial features sharp. Aim for 1024 px or more on the long edge.',
        'Vary expressions, clothing, lighting and backgrounds; avoid repeated burst shots.',
        'Use mostly eye-level photos. Extreme angles are optional, not the core of the set.',
        'Include back views if the intended videos show the subject from behind.',
        'Avoid beauty filters, heavy compression, watermarks and faces hidden by glasses or hair.',
        'Caption the visible angle, clothing, pose and background. Do not guess hidden features.',
        'Keep 2–3 additional photos out of training to judge likeness on unseen views.',
        'Photos teach appearance. To teach a distinctive action or motion, add video training separately.',
        'For objects, cover front, rear, both sides, three-quarter views, top and relevant details.'
    ],
    'workflow': ['Upload each photo with POST /media.',
                 'Create a dataset with POST /training/datasets.',
                 'Add each returned input file with POST /training/datasets/{id}/photos.',
                 'Review captions and coverage; POST /training/jobs to train.',
                 'Poll GET /training/jobs/{id}?wait=25.',
                 'POST /training/jobs/{id}/generate to make a video with the registered LoRA.'],
}


@contextmanager
def json_lock(path):
    if os.name!='nt':
        yield
        return
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateMutexW
    create.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR]
    create.restype=wintypes.HANDLE
    wait=kernel.WaitForSingleObject
    wait.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    release=kernel.ReleaseMutex
    release.argtypes=[wintypes.HANDLE]
    close=kernel.CloseHandle
    close.argtypes=[wintypes.HANDLE]
    key=hashlib.sha256(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()
    handle=create(None,False,'Local\\FreeVideoLoRA-'+key)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    acquired=False
    try:
        result=wait(handle,30000)
        if result not in (0,0x80):raise TimeoutError('Timed out waiting for training state')
        acquired=True
        yield
    finally:
        if acquired:release(handle)
        close(handle)


def write_json(path,value):
    with json_lock(path):
        _write_json(path,value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + secrets.token_hex(4) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt==7:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(.1*(attempt+1))


def read_json(path):
    with json_lock(path):
        return _read_json(path)


def _read_json(path):
    if os.name!='nt':
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                     wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    # Share deletion so a status reader never blocks the worker's atomic replacement.
    handle=create(str(Path(path)),0x80000000,7,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    with os.fdopen(descriptor,'r',encoding='utf-8-sig') as stream:
        return json.load(stream)


def identity(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix + r'[0-9a-f]{12}', value):
        raise ValueError('Invalid identity')
    return value


def contracts():
    from .schema import obj, string, number, enum, array
    ident = lambda prefix: string(13, pattern='^' + prefix + '[0-9a-f]{12}$')
    empty = obj()
    definitions = {}
    def add(method, path, body=None, prefix=None, query=None):
        definitions[(method, path)] = dict(body=body, query=query or empty, auth=True, local=True,
            path=obj({'id': ident(prefix)}, ('id',)) if prefix else None)
    add('GET', '/training')
    add('GET', '/training/guide')
    add('GET', '/training/datasets')
    add('POST', '/training/datasets', obj(dict(name=string(80), trigger=string(40, pattern=r'^[A-Za-z][A-Za-z0-9_]{2,39}$'),
        subject=string(80)), ('name', 'trigger', 'subject')))
    add('GET', '/training/datasets/{id}', prefix='d')
    photo = dict(file=string(1024), angle=enum(ANGLES), caption=string(2000),
                 source_sha256=string(64, pattern=r'^[0-9a-f]{64}$'))
    add('POST', '/training/datasets/{id}/photos', obj(photo, ('file', 'angle', 'caption')), 'd')
    add('PUT', '/training/datasets/{id}/captions', obj({'photos': array(obj(dict(id=ident('p'),
        caption=string(2000), angle=enum(ANGLES)), ('id', 'caption', 'angle')), high=200)}, ('photos',)), 'd')
    add('GET', '/training/jobs')
    add('POST', '/training/jobs', obj(dict(dataset=ident('d'), ref=string(100), epochs=number(1, 500, integer=True),
        rank=dict(type='integer', enum=[4,8,16]), megapixels=number(.1, 1)), ('dataset', 'ref')))
    add('GET', '/training/jobs/{id}', prefix='t', query=obj(dict(wait=number(0,25,default=0))))
    add('POST', '/training/jobs/{id}/cancel', empty, 't')
    add('POST', '/training/jobs/{id}/generate', obj(dict(prompt=string(20000), seconds=number(1.625,60),
        aspect=string(7), quality=enum(('light','medium','high','max')), strength=number(-4,4), ref=string(100)),
        ('prompt','seconds','ref')), 't')
    return definitions


class Training:
    def __init__(self, root, port):
        self.root = Path(root)
        self.home = self.root / 'agent' / 'training'
        self.home.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.lock = threading.RLock()

    def config(self):
        path = self.home / 'backend.json'
        return read_json(path) if path.is_file() else {}

    def status(self):
        config = self.config()
        missing = [k for k in ('python', 'repo', 'dit', 'text_encoder', 'vae', 'training_adapter')
                   if not config.get(k) or not Path(config[k]).exists()]
        ready_file = self.home / 'backend-ready.json'
        return dict(api_version=EXTENSION_VERSION, client_source_sha256=True,
                    ready=not missing and ready_file.is_file(), missing=missing,
                    backend='Fizgig MiniMax H3', installation=read_json(ready_file) if ready_file.is_file() else None,
                    running=[j['id'] for j in self.runs() if j['status'] in ('waiting', 'running', 'cancelling')],
                    hardware_note='8 GB training is experimental; CUDA out-of-memory is reported as failure, never as success.',
                    guide=GUIDE, setup_log=self.setup_log() if not ready_file.is_file() else None)

    def setup_log(self):
        path=self.home/'setup.log'
        if not path.is_file():return ''
        with path.open('rb') as stream:
            encoding='utf-16-le' if stream.read(2)==b'\xff\xfe' else 'utf-8'
            stream.seek(max(0,path.stat().st_size-2500))
            return stream.read().decode(encoding,errors='replace').lstrip('\ufeff')

    def dataset_path(self, name):
        return self.home / 'datasets' / identity(name, 'd') / 'dataset.json'

    def dataset(self, name):
        row = read_json(self.dataset_path(name))
        counts = {a: sum(p['angle'] == a for p in row['photos']) for a in ANGLES}
        return dict(row, coverage=counts, suggestions=[a for a in GUIDE['angles'] if not counts[a]],
                    recommended_count=20, ready_to_train=len(row['photos']) >= 5)

    def create_dataset(self, value):
        with self.lock:
            for path in (self.home / 'datasets').glob('*/dataset.json'):
                if read_json(path)['trigger'].lower() == value['trigger'].lower():
                    raise ValueError('This trigger word is already assigned to another dataset')
            name = 'd' + secrets.token_hex(6)
            row = dict(id=name, **value, photos=[], created_at=time.time())
            write_json(self.dataset_path(name), row)
            return self.dataset(name)

    def add_photo(self, name, value):
        from PIL import Image, ImageOps, UnidentifiedImageError
        source = self.port.input_path(value['file'])
        if source.stat().st_size > 25 * 1024**2:
            raise ValueError('Use photos up to 25 MB')
        data = source.read_bytes()
        source_digest = hashlib.sha256(data).hexdigest()
        if value.get('source_sha256') and value['source_sha256'] != source_digest:
            raise ValueError('Source photo hash does not match uploaded bytes')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError('Use a photo below 40 megapixels')
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((2048,2048))
                stream = io.BytesIO()
                image.save(stream, 'JPEG', quality=95)
                pixels = stream.getvalue()
                width, height = image.size
        except (UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError('Upload a readable photograph') from error
        digest = hashlib.sha256(pixels).hexdigest()
        with self.lock:
            row = self.dataset(name)
            if value.get('source_sha256'):
                previous = next((p for p in row['photos'] if p.get('source_sha256') == source_digest), None)
                if previous:
                    if previous['caption'] != self.caption(row['trigger'], value['caption']) or previous['angle'] != value['angle']:
                        raise ValueError('This source photo already exists with different caption or angle')
                    return row
            self.require_editable(name)
            if any(p['sha256'] == digest for p in row['photos']):
                raise ValueError('This photo is already in the dataset')
            if len(row['photos']) >= 200:
                raise ValueError('Use at most 200 photos per dataset')
            pid = 'p'+secrets.token_hex(6)
            folder = self.dataset_path(name).parent / 'photos'
            folder.mkdir(exist_ok=True)
            (folder / (pid+'.jpg')).write_bytes(pixels)
            caption = self.caption(row['trigger'], value['caption'])
            (folder / (pid+'.txt')).write_text(caption, encoding='utf-8')
            row['photos'].append(dict(id=pid, angle=value['angle'], caption=caption,
                width=width, height=height, sha256=digest, source_sha256=source_digest))
            self.save_dataset(row)
            return self.dataset(name)

    @staticmethod
    def caption(trigger, text):
        text = text.strip()
        if not text:
            raise ValueError('Describe what is visible in this photo')
        return text if re.search(r'\b'+re.escape(trigger)+r'\b', text) else trigger + ', ' + text

    def save_dataset(self, row):
        write_json(self.dataset_path(row['id']), {k: v for k,v in row.items()
            if k not in ('coverage','suggestions','recommended_count','ready_to_train')})

    def require_editable(self, name):
        if any(r['dataset']==name and r['status'] in ('waiting','running','cancelling') for r in self.runs()):
            raise ValueError('Wait for this dataset’s training run to finish before editing it')

    def captions(self, name, value):
        with self.lock:
            self.require_editable(name)
            row = self.dataset(name)
            known={p['id']: p for p in row['photos']}
            if any(p['id'] not in known for p in value['photos']):
                raise ValueError('Photo does not belong to this dataset')
            for p in value['photos']:
                target=known[p['id']]
                target.update(angle=p['angle'],caption=self.caption(row['trigger'],p['caption']))
                (self.dataset_path(name).parent/'photos'/(p['id']+'.txt')).write_text(target['caption'],encoding='utf-8')
            self.save_dataset(row)
            return self.dataset(name)

    def run_path(self, name):
        return self.home / 'runs' / identity(name,'t') / 'job.json'

    def run(self, name):
        import psutil
        path=self.run_path(name)
        row=read_json(path)
        if row.get('status') == 'succeeded' and row.get('lora'):
            try:
                filename = row['lora'].replace('\\', '/').split('/')[-1]
                row['adapter'] = inspect_adapter(Path(self.config()['lora_dir']) / filename)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row['adapter'] = dict(verified=False, error=str(error))
        if row['status'] in ('waiting','running','cancelling') and row.get('pid'):
            try:
                process=psutil.Process(row['pid'])
                alive=abs(process.create_time()-row['process_created'])<2 and process.is_running()
            except psutil.NoSuchProcess:
                alive=False
            if not alive:
                # Re-read in case the worker committed its result before exiting.
                row=read_json(path)
                if row['status'] in ('waiting','running','cancelling'):
                    row.update(status='interrupted',error='The training worker exited. Check its log before starting a new run.')
                    write_json(path,row)
        log=path.parent/'trainer.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-5000))
                row['log_tail']=stream.read().decode('utf-8',errors='replace')
        return row

    def runs(self):
        return [self.run(p.parent.name) for p in sorted((self.home/'runs').glob('*/job.json'))]

    def start(self, value):
        import psutil
        with self.lock:
            for row in self.runs():
                if row['ref']==value['ref']:
                    if row['request'] != value:
                        raise ValueError('Use a new ref for a different training request')
                    return row
            if any(r['status'] in ('waiting','running','cancelling') for r in self.runs()):
                raise ValueError('Another LoRA is training; wait or cancel that run')
            if any(self.port.snapshot()):
                raise ValueError('A video is queued or running; wait for it to finish')
            if not self.status()['ready']:
                raise ValueError('The training backend is not ready. GET /training lists missing prerequisites')
            dataset=self.dataset(value['dataset'])
            if len(dataset['photos'])<5:
                raise ValueError('Upload at least five distinct captioned photos; 20–30 are recommended')
            name='t'+secrets.token_hex(6)
            row=dict(id=name,dataset=dataset['id'],trigger=dataset['trigger'],ref=value['ref'],request=value,
                status='waiting',phase='starting',created_at=time.time(),pid=None,error=None,
                epochs=value.get('epochs',50),rank=value.get('rank',8),megapixels=value.get('megapixels',.25))
            path=self.run_path(name)
            write_json(path,row)
            env=os.environ.copy()
            env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
            env['FREEVIDEO_HOME']=str(self.root)
            env['PYTHONUTF8']='1'
            env['PATH']=os.pathsep.join(p for p in env.get('PATH','').split(os.pathsep) if 'openai'+os.sep+'codex' not in p.lower())
            try:
                with (path.parent/'worker.log').open('ab') as log:
                    process=subprocess.Popen([sys.executable,'-u','-B','-m','freevideo_engine.agent.training_worker',
                        '--root',str(self.root),'--job',name],env=env,stdout=log,stderr=subprocess.STDOUT)
                # Worker waits for ownership to be recorded before modifying the job.
                row.update(pid=process.pid,process_created=psutil.Process(process.pid).create_time())
            except OSError as error:
                row.update(status='failed',error='Could not launch trainer: '+str(error),finished_at=time.time())
            write_json(path,row)
            return row

    def cancel(self,name):
        with self.lock:
            row=self.run(name)
            if row['status'] in ('waiting','running','cancelling'):
                (self.run_path(name).parent/'cancel').touch()
            return row


def register(routes, guarded, reply, body, port, jobs):
    from ..comfy_bridge import installation_root
    service=Training(installation_root(),port)

    async def handler(request):
        method=request.method
        path=request.match_info.route.resource.canonical.removeprefix(PREFIX)
        name=request.match_info.get('id')
        value=await body(request) if method in ('POST','PUT') else None
        try:
            if path=='/training': result=await asyncio.to_thread(service.status)
            elif path=='/training/guide': result=GUIDE
            elif path=='/training/datasets':
                result=await asyncio.to_thread(service.create_dataset,value) if method=='POST' else dict(datasets=[
                    service.dataset(p.parent.name) for p in (service.home/'datasets').glob('*/dataset.json')])
            elif path=='/training/datasets/{id}': result=await asyncio.to_thread(service.dataset,name)
            elif path.endswith('/photos'): result=await asyncio.to_thread(service.add_photo,name,value)
            elif path.endswith('/captions'): result=await asyncio.to_thread(service.captions,name,value)
            elif path=='/training/jobs':
                result=await asyncio.to_thread(service.start,value) if method=='POST' else dict(jobs=await asyncio.to_thread(service.runs))
            elif path.endswith('/cancel'): result=await asyncio.to_thread(service.cancel,name)
            elif path.endswith('/generate'):
                row=await asyncio.to_thread(service.run,name)
                if row['status']!='succeeded': raise ValueError('Train a successful adapter before generating')
                if not row.get('adapter', {}).get('verified'): raise ValueError('Registered adapter is missing or invalid')
                prompt=value['prompt']
                if row['trigger'] not in prompt: prompt=row['trigger']+', '+prompt
                spec=dict(prompt=prompt,seconds=value['seconds'],aspect=value.get('aspect','16:9'),
                    quality=value.get('quality','light'),megapixels=.5,label='Trained subject',
                    loras=[dict(name=row['lora'],strength=value.get('strength',.8))],ref=value['ref'])
                rows,replay=await asyncio.to_thread(jobs.submit_request,spec,client=request.get('agent_client'))
                await jobs.tick()
                result=dict(job=jobs.view(rows[0]))
            else:
                deadline=time.monotonic()+min(25,float(request.query.get('wait',0)))
                while True:
                    result=await asyncio.to_thread(service.run,name)
                    if result['status'] not in ('waiting','running','cancelling') or time.monotonic()>=deadline: break
                    await asyncio.sleep(.5)
            return reply(result)
        except (ValueError,FileNotFoundError) as error:
            return reply(dict(error=dict(code='training_invalid',message=str(error))),400)
    for method,path in contracts():
        routes.route(method,PREFIX+path)(guarded(handler))
    return service
