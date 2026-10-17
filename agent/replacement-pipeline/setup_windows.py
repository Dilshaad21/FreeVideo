"""Isolated, explicit replacement-runtime setup; never modifies existing envs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

WAN_COMMIT='fde3c7e10c47c854913d5877d955901a35e44714'
SAM_COMMIT='2b90b9f5ceec907a1c18123530e92e794ad901a4'


def download(url,path):
    from prefetch import fetch
    path=Path(path)
    fetch(url,path,path.parent/'asset-progress.json')


def source(repo,commit,target):
    target=Path(target)
    if (target/'PINNED_COMMIT').is_file():
        if (target/'PINNED_COMMIT').read_text()!=commit:raise RuntimeError('Source revision conflict')
        return
    archive=target.parent/(target.name+'.zip')
    download('https://codeload.github.com/'+repo+'/zip/'+commit,archive)
    temporary=target.parent/(target.name+'-extract')
    if temporary.exists():shutil.rmtree(temporary)
    temporary.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            dest=(temporary/member.filename).resolve()
            if not dest.is_relative_to(temporary.resolve()):raise ValueError('Unsafe source archive')
        bundle.extractall(temporary)
    extracted=next(temporary.iterdir())
    if target.exists():raise RuntimeError('Unrecognized existing source folder')
    extracted.rename(target)
    temporary.rmdir()
    (target/'PINNED_COMMIT').write_text(commit)
    archive.unlink()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--models',action='store_true')
    parser.add_argument('--skip-install',action='store_true',help='Recheck an already prepared environment without pip changes')
    args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('Windows setup only')
    root=Path(args.root).resolve()
    home=root/'agent/replacement'
    home.mkdir(parents=True,exist_ok=True)
    from replacement_support import file_lock,atomic_json,bind_worker_tree
    setup_lock=file_lock(home/'setup.lock',timeout=0);setup_lock.__enter__()
    state=home/'setup.json'
    retained_models=json.loads(state.read_text()).get('models',False) if state.exists() else False
    def update(**kw):
        row=json.loads(state.read_text()) if state.exists() else {}
        row.update(kw)
        atomic_json(state,row)
    update(status='running',phase='source',runtime_imports=False,models=retained_models,error=None)
    try:
        setup_job_handle=bind_worker_tree()
        required_free=(8 if args.skip_install else 35)*1024**3
        if shutil.disk_usage(home).free<required_free:raise RuntimeError('Insufficient free disk for setup: '+str(required_free//1024**3)+' GB required')
        source('deepbeepmeep/WanGP',WAN_COMMIT,home/'WanGP')
        source('facebookresearch/sam2',SAM_COMMIT,home/'sam2')
        trainer=root/'agent/training/env'
        python=home/'env/Scripts/python.exe'
        if not python.exists():
            subprocess.run([str(trainer/'Scripts/python.exe'),'-m','venv',str(home/'env')],check=True)
        site=home/'env/Lib/site-packages'
        # Read-only fallback for the large tested torch/CUDA stack. New dependencies
        # install into this env and shadow fallback packages, leaving both old envs intact.
        (site/'freevideo_trainer_fallback.pth').write_text(str(trainer/'Lib/site-packages')+'\n')
        (site/'freevideo_sam_source.pth').write_text(str(home/'sam2')+'\n')
        update(phase='dependencies')
        packages=['pip','wheel','setuptools','gradio==5.29.0','gradio_rangeslider','optimum-quanto==0.2.7',
            'diffusers==0.36.0','transformers==4.54.0','numpy==2.1.2','opencv-python>=4.12.0.88',
            'comfy-kitchen==0.2.35','accelerate','sentencepiece','ftfy','einops','imageio','imageio-ffmpeg',
            'av','decord','scipy','timm','iopath','hydra-core','omegaconf','huggingface_hub[hf_xet]',
            'hf_xet>=1.5.2','orjson==3.11.6','ffmpeg-python','moviepy==1.0.3','mutagen','soundfile',
            'sounddevice','pygame','librosa==0.11.0','pyloudnorm','nvidia-ml-py','gguf==0.17.1',
            'rotary-embedding-torch','loguru','easydict','num2words','open_clip_torch','piexif','mcp==1.30.0',
            'dashscope','keyring','apprise','markdown','pydantic==2.11.10','resvg-py','xxhash','gitpython','psutil',
            'ultralytics','pycocotools','onnxruntime-gpu==1.22.0','rembg==2.0.65','segment-anything',
            'openai-whisper==20250625','tiktoken','torchaudio','einshape==1.0',
            'smplfitter @ https://github.com/deepbeepmeep/smplfitter/releases/download/v0.2.10/smplfitter-0.2.10-py3-none-any.whl',
            'chumpy @ https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl']
        # Do not let dependency resolution replace the shared torch/CUDA build.
        constraints=home/'constraints.txt'
        versions=subprocess.check_output([str(trainer/'Scripts/python.exe'),'-c',
            "import importlib.metadata as m;print('\\n'.join(x+'=='+m.version(x) for x in ['torch','torchvision']));print('torchaudio=='+m.version('torch'))"],text=True)
        constraints.write_text(versions)
        env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUTF8='1',SAM2_BUILD_CUDA='0',HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='120',HF_HUB_ETAG_TIMEOUT='30')
        env['PATH']=os.pathsep.join(p for p in env.get('PATH','').split(os.pathsep)
            if '/openai/codex' not in p.replace('\\','/').lower())
        if not args.skip_install:
            subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--upgrade',
                            '--extra-index-url','https://download.pytorch.org/whl/cu128',
                            '-c',str(constraints)]+packages,check=True,env=env)
        update(phase='probe')
        probe=Path(__file__).parent/'runtime.py'
        subprocess.run([str(python),str(probe),'probe','--home',str(home)],env=env,check=True,timeout=7200)
        snapshot=subprocess.check_output([str(python),'-m','pip','freeze'],text=True,env=env)
        (home/'dependencies.txt').write_text(snapshot)
        update(runtime_imports=True)
        config=dict(python=str(python),home=str(home),source=str(Path(__file__).parent.resolve()),
                    wangp=str(home/'WanGP'),sam2=str(home/'sam2'),sam_config='configs/sam2.1/sam2.1_hiera_t.yaml',
                    sam_checkpoint=str(home/'models/sam2.1_hiera_tiny.pt'),profile='5',attention='sdpa',
                    working_height=480,steps=20,window=81,max_temperature_c=87,
                    min_disk_bytes=8*1024**3,min_ram_bytes=2*1024**3,ram_floor_bytes=512*1024**2)
        (home/'backend.json').write_text(json.dumps(config,indent=2))
        if args.models:
            update(phase='models')
            download('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',config['sam_checkpoint'])
            subprocess.run([str(python),str(probe),'models','--home',str(home)],env=env,check=True,timeout=7200)
            update(models=True)
        update(status='succeeded',phase='ready' if args.models or retained_models else 'runtime-ready')
    except BaseException as error:
        update(status='failed',error=str(error))
        raise
    finally:setup_lock.__exit__(None,None,None)


if __name__=='__main__':main()
