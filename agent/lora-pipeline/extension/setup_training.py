"""Install a pinned, isolated training runtime; never change FreeVideo's inference packages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root)
    home=root/'agent'/'training'
    config=json.loads((home/'backend.json').read_text(encoding='utf-8'))
    env=os.environ.copy()
    env.update(HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1',DISABLE_CUDA='1',PYTHONUTF8='1')
    env['PATH']=os.pathsep.join(x for x in env.get('PATH','').split(os.pathsep) if 'openai'+os.sep+'codex' not in x.lower())
    def run(cmd): subprocess.run([str(x) for x in cmd],env=env,check=True)
    uv=root/'tools'/'uv.exe'
    venv=Path(config['python']).parents[1]
    if not Path(config['python']).is_file():run([uv,'venv','--python',sys.executable,venv])
    packages=['torch==2.10.0','torchvision==0.25.0','accelerate==1.6.0','diffusers==0.32.1',
        'transformers==4.57.6','safetensors==0.5.3','einops==0.7.0','bitsandbytes==0.48.2',
        'comfy-kitchen==0.2.31','toml==0.10.2','voluptuous==0.15.2','pillow==12.3.0',
        'numpy>=1.26.4,<3','opencv-python==4.10.0.84','tensorboard==2.20.0','tqdm==4.67.1',
        'huggingface-hub==0.34.3','hf-xet>=1.0','pyyaml==6.0.3','packaging==25.0',
        'psutil>=5.9','nvidia-ml-py>=12.0','sentencepiece==0.2.1','timm>=0.9.0',
        'soundfile>=0.12,<1','imageio-ffmpeg>=0.5']
    run([uv,'pip','install','--python',config['python'],'--extra-index-url',
         'https://download.pytorch.org/whl/cu128','--index-strategy','unsafe-best-match',*packages])
    # The actual upstream CLI parsers are part of the readiness check.
    env['PYTHONPATH']=str(Path(config['repo'])/'src')
    for script in ('cache.py','train.py'):
        run([config['python'],Path(config['repo'])/'src'/'fizgig'/'families'/script,'--help'])
    download=r'''
from huggingface_hub import hf_hub_download, HfApi
import json, pathlib
home=pathlib.Path(__import__('sys').argv[1]); config=json.loads((home/'backend.json').read_text())
files={'dit':('Comfy-Org/MiniMax-H3','diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors'),
       'vae':('Comfy-Org/MiniMax-H3','vae/minimax_h3_video_vae_fp16.safetensors'),
       'training_adapter':('circlestone-labs/MiniMax-H3-Image-Training-Adapter','minimax_h3_image_training_adapter.safetensors')}
revisions={}
for key,(repo,name) in files.items():
 if pathlib.Path(config[key]).is_file():continue
 sha=revisions.setdefault(repo,HfApi().repo_info(repo).sha)
 print('Downloading',key,repo,name,sha,flush=True)
 actual=hf_hub_download(repo,name,revision=sha,local_dir=str(home/'models'/key))
 config[key]=actual
 (home/'backend.json').write_text(json.dumps(config,indent=2))
(home/'model-revisions.json').write_text(json.dumps(revisions,indent=2))
'''
    run([config['python'],'-c',download,home])
    probe=r'''
import torch
from fizgig.families.registry import get
from fizgig.families import train
from fizgig.minimax.driver import MiniMaxDriver
assert torch.cuda.is_available(), 'CUDA training device is unavailable'
x=torch.ones((32,32),device='cuda'); assert (x@x).sum().item()==32768
assert get('minimax').training_ready
print(torch.cuda.get_device_name(0),flush=True)
'''
    run([config['python'],'-c',probe])
    (home/'backend-ready.json').write_text(json.dumps(dict(ready=True,checked_at=time.time(),
        trainer_commit=config['trainer_commit'],cuda_smoke_test=True),indent=2),encoding='utf-8')
    print('Training backend ready',flush=True)


if __name__=='__main__':main()
