"""Run the trainer in its own environment while holding FreeVideo's GPU lease."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .training import identity, read_json, write_json, inspect_adapter
from ..locking import runtime_lock


class Cancelled(Exception):
    pass


def build_stages(config, row, folder, dataset):
    python=config['python']
    cache=str(Path(config['repo'])/'src'/'fizgig'/'families'/'cache.py')
    train=str(Path(config['repo'])/'src'/'fizgig'/'families'/'train.py')
    toml=folder/'dataset.toml'
    common=[python,'-u',cache,'--family','minimax','--dataset_config',str(toml),'--num_workers','1','--batch_size','1']
    name='fv_'+row['dataset']+'_'+row['id']
    command=[python,'-u',train,'--family','minimax','--dit',config['dit'],
        '--dataset_config',str(toml),'--output_dir',str(folder/'checkpoints'),'--output_name',name,
        '--precision','auto','--blocks_to_swap','-1','--network_dim',str(row['rank']),
        '--network_alpha',str(row['rank']),'--learning_rate','1e-6','--optimizer_type','automagic3',
        '--max_train_epochs',str(row['epochs']),'--save_every_n_epochs',str(min(10,row['epochs'])),
        '--seed','42','--training_adapter',config['training_adapter'],
        '--family_option','photo_blocks=20-49','--family_option','lownoise_pct=60',
        '--family_option','caption_dropout=0.05','--compile_blocks','off',
        '--save_state','--save_state_on_train_end','--keep_last_n_states','2',
        '--trigger_word',dataset['trigger'],'--metadata_trigger_phrase',dataset['trigger']]
    return name,[('latents',common+['--stage','latents','--model',config['vae'],'--skip_existing']),
                 ('text',common+['--stage','text','--model',config['text_encoder']]),('training',command)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    home=root/'agent'/'training'
    folder=home/'runs'/identity(args.job,'t')
    path=folder/'job.json'
    for _ in range(100):
        row=read_json(path)
        if row.get('pid') is not None: break
        time.sleep(.1)
    else: raise RuntimeError('Worker ownership was not recorded')
    # Windows venv launchers may spawn a second process; track the interpreter that runs this job.
    import psutil
    row.update(pid=os.getpid(),process_created=psutil.Process(os.getpid()).create_time())
    def update(**value):
        row.update(value,heartbeat_at=time.time())
        write_json(path,row)
    def cancelled():
        if (folder/'cancel').exists(): raise Cancelled()
    try:
        config=read_json(home/'backend.json')
        dataset=read_json(home/'datasets'/identity(row['dataset'],'d')/'dataset.json')
        # Freeze this run's dataset; later runs can change captions without rewriting history.
        photos=folder/'photos'
        shutil.copytree(home/'datasets'/row['dataset']/'photos',photos)
        write_json(folder/'dataset.json',dataset)
        side=max(256,int((row['megapixels']*1_000_000)**.5)//32*32)
        toml='[general]\nresolution = ['+str(side)+', '+str(side)+']\ncaption_extension = ".txt"\nbatch_size = 1\nnum_repeats = 1\nenable_bucket = true\nbucket_no_upscale = true\n\n[[datasets]]\nimage_directory = '+json.dumps(photos.as_posix())+'\ncache_directory = '+json.dumps((folder/'cache').as_posix())+'\n'
        (folder/'dataset.toml').write_text(toml,encoding='utf-8')
        name,stages=build_stages(config,row,folder,dataset)
        write_json(folder/'commands.json',stages)
        env=os.environ.copy()
        env.update(PYTHONPATH=str(Path(config['repo'])/'src'),PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',
            PYTHONFAULTHANDLER='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
        # The trainer is an independent child, not an inference worker with an inherited lease.
        for key in list(env):
            if (key.startswith(('FREEVIDEO_RUNTIME_LOCK_', 'FREEVIDEO_BOOTSTRAP_', 'COMFY'))
                    or key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTHONHOME')):
                env.pop(key,None)
        env['PATH'] = os.pathsep.join([str(Path(config['python']).parent)] + [
            entry for entry in env.get('PATH', '').split(os.pathsep)
            if not any(part in entry.replace('\\', '/').lower()
                       for part in ('/envs/comfyui-', '/envs/unified'))])
        with ExitStack() as stack:
            while True:
                cancelled()
                try:
                    stack.enter_context(runtime_lock(root/'engine.lock',inherit=False))
                    break
                except BlockingIOError:
                    update(status='waiting',phase='waiting-for-gpu')
                    time.sleep(1)
            update(status='running',started_at=time.time())
            for phase,command in stages:
                cancelled()
                update(phase=phase)
                with (folder/'trainer.log').open('ab') as log:
                    log.write(('\nSTAGE '+phase+'\n').encode()); log.flush()
                    stage_env=dict(env)
                    if phase=='text' and config.get('text_disk_streaming'):
                        stage_env.update(FIZGIG_TE_RAM_OK='1',FIZGIG_TE_DISK_STREAM='1')
                    elif phase=='text' and config.get('text_streaming'):
                        stage_env['FIZGIG_TE_RAM_OK']='1'
                    child=subprocess.Popen(command,cwd=config['repo'],env=stage_env,stdout=log,stderr=subprocess.STDOUT)
                    try:
                        while child.poll() is None:
                            cancelled()
                            update(child_pid=child.pid)
                            time.sleep(1)
                    except BaseException:
                        import psutil
                        try:
                            process=psutil.Process(child.pid)
                            children=process.children(recursive=True)
                            for p in children: p.terminate()
                            process.terminate()
                            _,alive=psutil.wait_procs(children+[process],timeout=5)
                            for p in alive: p.kill()
                        except psutil.NoSuchProcess: pass
                        child.wait()
                        raise
                    if child.returncode:
                        tail=(folder/'trainer.log').read_text(encoding='utf-8',errors='replace')[-5000:]
                        raise RuntimeError(f'{phase} failed (exit {child.returncode}).\n'+tail)
            checkpoint=folder/'checkpoints'/(name+'.safetensors')
            if not checkpoint.is_file(): raise RuntimeError('Trainer exited without its final adapter')
            inspect_adapter(checkpoint)
            target=Path(config['lora_dir'])/(name+'.safetensors')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.part')
            shutil.copyfile(checkpoint,temporary)
            os.replace(temporary,target)
            update(status='succeeded',phase='done',finished_at=time.time(),lora=str(Path('trained')/target.name),
                checkpoint_bytes=target.stat().st_size,error=None)
    except Cancelled:
        update(status='cancelled',phase='cancelled',finished_at=time.time())
    except Exception as error:
        update(status='failed',phase=row.get('phase'),error=str(error),finished_at=time.time())
        raise


if __name__=='__main__':
    main()
