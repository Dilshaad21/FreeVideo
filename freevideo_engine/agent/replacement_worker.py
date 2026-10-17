"""Execute verified, resumable stages under FreeVideo's GPU lease."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import subprocess
import time

from .training import identity
from .replacement_support import (atomic_json,read_json,mutate,digest,execution_signature,
    bind_worker_tree,terminate_tree,Journal,ResourceMonitor)
from ..locking import runtime_lock


class Cancelled(Exception):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();root=Path(args.root);home=root/'agent/replacement'
    folder=home/'runs'/identity(args.job,'r');state=folder/'job.json'
    for _ in range(300):
        row=read_json(state)
        if row.get('launch_pid'):break
        time.sleep(.1)
    else:
        mutate(state,lambda row:row.update(status='failed',error_code='launch_timeout',error='Worker ownership was not recorded'))
        return
    import psutil
    def update(**kw):
        nonlocal row
        row=mutate(state,lambda current:current.update(kw,heartbeat_at=time.time()))
    def cancel():
        if (folder/'cancel').exists():raise Cancelled()
    update(pid=os.getpid(),process_created=psutil.Process().create_time())
    monitor=None
    try:
        # The OS closes this raw handle on worker exit, killing GPU descendants.
        worker_job_handle=bind_worker_tree()
        config=read_json(home/'backend.json')
        signature=execution_signature(config)
        if row.get('execution_signature')!=signature:raise ValueError('Pipeline/configuration changed; start a new ref')
        journal=Journal(folder,signature);monitor=ResourceMonitor(config)
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONUNBUFFERED='1',
                 TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
                 HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='120',HF_HUB_ETAG_TIMEOUT='30')
        for key in list(env):
            if key.startswith(('FREEVIDEO_RUNTIME_LOCK_','FREEVIDEO_BOOTSTRAP_','COMFY')) or key in (
                    'PYTHONHOME','PYTHONPATH','PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):
                env.pop(key,None)
        env['PATH']=os.pathsep.join([str(Path(config['python']).parent)]+[
            p for p in env.get('PATH','').split(os.pathsep)
            if not any(bad in p.replace('\\','/').lower() for bad in ('/envs/comfyui-','/envs/unified','/openai/codex'))])
        def stage(name,action,request,outputs):
            cancel();update(phase=name)
            key=journal.key(action,request)
            effective=list(outputs)
            if action=='generate' and (folder/'generated.json').exists():
                try:receipt=read_json(folder/'generated.json')
                except ValueError:receipt={}
                if receipt.get('video'):effective.append(Path(receipt['video']))
            if journal.reusable(name,key,effective):
                update(reused_stages=row.get('reused_stages',[])+[name]);return
            # A successful generation receipt survives a crash before its journal commit.
            if action=='generate' and (folder/'generated.json').exists():
                try:receipt=read_json(folder/'generated.json')
                except ValueError:receipt={}
                video=Path(receipt.get('video',''))
                if (receipt.get('phase_key')==key and video.is_file() and not video.is_symlink()
                        and video.resolve().is_relative_to((home/'generated').resolve())
                        and digest(video)==receipt.get('sha256')):
                    journal.complete(name,key,outputs+[video]);return
            for output in outputs:
                output=Path(output)
                if not output.resolve().is_relative_to(folder.resolve()):raise ValueError('Refusing to delete an external stage output')
                if output.is_dir() and not output.is_symlink():shutil.rmtree(output)
                else:output.unlink(missing_ok=True)
            request_path=folder/(name+'.json');atomic_json(request_path,dict(request,_phase_key=key))
            timeout=config.get('phase_timeout_seconds',{}).get(name,7200 if action=='generate' else 900)
            started=time.monotonic()
            with (folder/'runtime.log').open('ab') as log:
                log.write(('\nSTAGE '+name+'\n').encode());log.flush()
                child=subprocess.Popen([config['python'],'-u',str(Path(config['source'])/'runtime.py'),action,
                    '--home',str(home),'--request',str(request_path)],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:created=psutil.Process(child.pid).create_time()
                except psutil.NoSuchProcess:created=None
                update(child_pid=child.pid,child_created=created)
                try:
                    while child.poll() is None:
                        cancel()
                        if time.monotonic()-started>timeout:raise TimeoutError(name+' exceeded its configured time limit')
                        update(resources=monitor.sample());time.sleep(1)
                    if child.returncode:
                        with (folder/'runtime.log').open('rb') as stream:
                            stream.seek(max(0,(folder/'runtime.log').stat().st_size-2500))
                            tail=stream.read().decode(errors='replace')
                        raise RuntimeError(name+' failed: '+tail)
                except BaseException:
                    terminate_tree(child.pid,created)
                    if child.poll() is None:child.kill()
                    child.wait();raise
                finally:update(child_pid=None,child_created=None)
            if action=='generate':outputs=outputs+[Path(read_json(folder/'generated.json')['video'])]
            journal.complete(name,key,outputs)
            durations=dict(row.get('stage_seconds',{}));durations[name]=round(time.monotonic()-started,2)
            update(stage_seconds=durations)
        with ExitStack() as stack:
            while True:
                cancel()
                try:stack.enter_context(runtime_lock(root/'engine.lock',inherit=False));break
                except BlockingIOError:update(phase='waiting-for-gpu');time.sleep(1)
            for partial in folder.glob('.output-*.partial'):
                if partial.is_dir() and not partial.is_symlink():shutil.rmtree(partial)
            if shutil.disk_usage(folder).free<config.get('min_disk_bytes',8*1024**3):
                raise RuntimeError('resource_limit: need at least 8 GB free disk before starting')
            if psutil.virtual_memory().available<config.get('min_ram_bytes',2*1024**3):
                raise RuntimeError('resource_limit: need at least 2 GB available RAM before starting')
            update(status='running',started_at=row.get('started_at',time.time()),resources=monitor.sample(force=True),reused_stages=[])
            inputs=folder/'inputs';inputs.mkdir(exist_ok=True)
            frozen={}
            for label,key in [('video','video_path'),('character','character_path')]:
                cancel();source=Path(row[key]);target=inputs/(label+source.suffix)
                expected=row[label+'_sha256']
                if not target.exists() or digest(target)!=expected:
                    if digest(source)!=expected:raise ValueError('Input changed after submission')
                    temporary=target.with_suffix(target.suffix+'.tmp');shutil.copyfile(source,temporary)
                    if digest(temporary)!=expected:raise ValueError('Input changed during copy')
                    temporary.replace(target)
                frozen[key]=str(target)
            update(**frozen)
            stage('prepare','prepare',dict(source=row['video_path'],character=row['character_path'],folder=str(folder)),
                  [folder/'originals',folder/'source-frames',folder/'media.json',folder/'control.mkv',folder/'reference.png'])
            meta=read_json(folder/'media.json')
            stage('track-source','track',dict(frames=str(folder/'source-frames'),destination=str(folder/'source-masks'),box=row['box']),[folder/'source-masks'])
            stage('trim-source-masks','trim-masks',dict(source=str(folder/'source-masks'),destination=str(folder/'source-masks-trimmed'),
                frames=meta['frames']),[folder/'source-masks-trimmed'])
            stage('mask-video','mask-video',dict(masks=str(folder/'source-masks'),fps=meta['fps'],output=str(folder/'mask.mkv')),[folder/'mask.mkv'])
            request=dict(prompt=row['prompt'],character=str(folder/'reference.png'),resolution=meta['resolution'],
                frames=meta['padded_frames'],fps=meta['fps'],video=str(folder/'control.mkv'),mask=str(folder/'mask.mkv'),result=str(folder/'generated.json'))
            # stage cleanup never removes external generated files; the immutable receipt owns them.
            stage('generate','generate',request,[folder/'generated.json'])
            generated=read_json(folder/'generated.json')['video']
            stage('extract-generated','extract',dict(video=generated,destination=str(folder/'generated-frames'),
                png=str(folder/'generated-png'),frames=meta['frames']),[folder/'generated-frames',folder/'generated-png'])
            stage('track-generated','track',dict(frames=str(folder/'generated-frames'),destination=str(folder/'generated-masks'),box=row['box']),[folder/'generated-masks'])
            journal.verify_all()
            if digest(row['video_path'])!=row['video_sha256']:raise ValueError('Frozen source video changed')
            stage('composite','composite',dict(source_dir=str(folder/'originals'),generated_dir=str(folder/'generated-png'),
                source_masks=str(folder/'source-masks-trimmed'),generated_masks=str(folder/'generated-masks'),original=row['video_path'],
                destination=str(folder/'output'),fps=meta['fps']),[folder/'output'])
            cancel();report=read_json(folder/'output/verification.json')
            if not report.get('verified'):raise RuntimeError('Output did not pass preservation checks')
            update(status='succeeded',phase='complete',finished_at=time.time(),verification=report,
                resources=monitor.summary,preview_url='/freevideo/agent/v1/replacement/jobs/'+row['id']+'/preview',
                files={name:'/freevideo/agent/v1/replacement/jobs/'+row['id']+'/files/'+name
                       for name in ('master.mkv','preview.mp4','verification.json')})
    except Cancelled:update(status='cancelled',phase='cancelled',finished_at=time.time())
    except BaseException as error:
        message=str(error);lower=message.lower()
        code=('out_of_memory' if 'out of memory' in lower else 'resource_limit' if 'resource_limit:' in lower
              else 'phase_timeout' if isinstance(error,TimeoutError) else 'replacement_failed')
        update(status='failed',error=message,error_code=code,finished_at=time.time(),resources=monitor.summary if monitor else {})
        raise


if __name__=='__main__':main()
