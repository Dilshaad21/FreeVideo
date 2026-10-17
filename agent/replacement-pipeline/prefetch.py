"""Resumable, locked weight downloads with validated range and content receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile

from replacement_support import atomic_json,digest,file_lock,read_json


def validate_artifact(path):
    path=Path(path)
    # The temporary suffix must not change the artifact's format check.
    suffix=path.with_suffix('').suffix if path.suffix=='.part' else path.suffix
    if suffix=='.safetensors':
        with path.open('rb') as stream:
            prefix=stream.read(8)
            if len(prefix)!=8:raise ValueError('Truncated safetensors header')
            size=struct.unpack('<Q',prefix)[0]
            if not 2<=size<=64*1024**2:raise ValueError('Invalid safetensors header length')
            header=json.loads(stream.read(size))
        tensors=[row for key,row in header.items() if key!='__metadata__']
        if not tensors:raise ValueError('No tensors in checkpoint')
        length=max(row['data_offsets'][1] for row in tensors)
        if path.stat().st_size!=8+size+length:raise ValueError('Checkpoint payload length does not match its header')
    elif suffix in ('.pt','.zip'):
        with zipfile.ZipFile(path) as bundle:
            if bundle.testzip() is not None:raise ValueError('Corrupt checkpoint/archive ZIP')
    elif path.stat().st_size==0:raise ValueError('Empty downloaded asset')


def promote(partial,path,url,expected_sha256=None):
    validate_artifact(partial)
    actual=digest(partial)
    if expected_sha256 and actual!=expected_sha256:raise ValueError('Downloaded SHA256 does not match the pinned checksum')
    partial.replace(path)
    stat=path.stat()
    atomic_json(path.with_suffix(path.suffix+'.download.json'),dict(url=url,sha256=actual,
        bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))


def fetch(url,path,state,expected_sha256=None):
    path=Path(path);state=Path(state);path.parent.mkdir(parents=True,exist_ok=True)
    with file_lock(path.with_suffix(path.suffix+'.download.lock'),timeout=0):
        if path.is_symlink():raise ValueError('Refusing a symlinked asset')
        receipt_path=path.with_suffix(path.suffix+'.download.json')
        if path.exists():
            receipt=read_json(receipt_path) if receipt_path.exists() else {}
            actual=digest(path)
            if (receipt.get('url',url)!=url or receipt.get('sha256',actual)!=actual
                    or expected_sha256 and expected_sha256!=actual):raise ValueError('Existing asset failed checksum verification: '+path.name)
            validate_artifact(path)
            stat=path.stat();atomic_json(receipt_path,dict(url=url,sha256=actual,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
            return
        partial=path.with_suffix(path.suffix+'.part')
        if partial.is_symlink():raise ValueError('Refusing a symlinked partial download')
        previous=partial.stat().st_size if partial.exists() else 0
        headers={'Range':f'bytes={previous}-'} if previous else {}
        try:response=urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120)
        except urllib.error.HTTPError as error:
            match=re.fullmatch(r'bytes \*/(\d+)',error.headers.get('Content-Range',''))
            if error.code==416 and previous and match and previous==int(match[1]):
                promote(partial,path,url,expected_sha256);return
            raise
        with response as source:
            resume=previous and source.status==206
            if source.status==206:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',source.headers.get('Content-Range',''))
                if not match or int(match[1])!=previous or int(match[2])<previous or int(match[2])+1!=int(match[3]):
                    raise ValueError('Server returned an inconsistent resume range')
                expected=int(match[3])
            elif source.status==200:
                expected=int(source.headers.get('Content-Length',0))
                if not expected:raise ValueError('Asset server must supply a content length')
            else:raise ValueError('Unexpected asset response status')
            if 'text/html' in source.headers.get('Content-Type',''):raise ValueError('Asset URL returned HTML')
            done=previous if resume else 0;mode='ab' if resume else 'wb';last=0
            with partial.open(mode) as output:
                while True:
                    block=source.read(4*1024*1024)
                    if not block:break
                    output.write(block);done+=len(block)
                    if done>expected:raise ValueError('Asset exceeded its declared size')
                    if time.monotonic()-last>15:
                        value=dict(status='downloading',file=path.name,bytes=done,total=expected)
                        atomic_json(state,value);print(json.dumps(value),flush=True);last=time.monotonic()
            if done!=expected:raise ValueError('Incomplete asset download; partial bytes retained for resume')
        promote(partial,path,url,expected_sha256)


def install_wangp_downloader(downloads,state):
    """Use WanGP's downloader extension points inside the isolated setup process."""
    from urllib.parse import quote
    def direct(url,filename,gen=None,show_filename=True):
        fetch(url.split('|')[0],Path(filename),state)
        return str(filename)
    def hub(*,gen=None,show_filename=True,file_index=1,file_count=1,**kwargs):
        if set(kwargs)-{'repo_id','filename','local_dir','subfolder'}:raise ValueError('Unsupported WanGP asset download arguments')
        root=Path(kwargs['local_dir']).resolve()
        relative=Path(kwargs.get('subfolder') or '')/kwargs['filename']
        target=(root/relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(root):raise ValueError('Unsafe asset path')
        url='https://huggingface.co/'+quote(kwargs['repo_id'],safe='/')+'/resolve/main/'+quote(relative.as_posix(),safe='/')
        fetch(url,target,state)
        return str(target)
    downloads.download_file=direct;downloads._hf_download=hub


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();home=Path(args.root)/'agent/replacement';home.mkdir(parents=True,exist_ok=True)
    state=home/'prefetch.json'
    with file_lock(home/'prefetch.lock',timeout=0):
        try:
            for url,path in [
                ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',home/'models/sam2.1_hiera_tiny.pt'),
                ('https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_animate_14B_quanto_bf16_int8.safetensors',
                 home/'WanGP/ckpts/wan2.2_animate_14B_quanto_bf16_int8.safetensors'),
                ('https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/models_t5_umt5-xxl-enc-quanto_int8.safetensors',
                 home/'WanGP/ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors')]:fetch(url,path,state)
            atomic_json(state,dict(status='succeeded',note='Core weights verified; supporting runtime validation is separate'))
        except BaseException as error:
            atomic_json(state,dict(status='failed',error=str(error)));raise


if __name__=='__main__':main()
