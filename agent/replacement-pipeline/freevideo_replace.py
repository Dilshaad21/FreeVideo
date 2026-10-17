"""Compact, persistent client for FreeVideo's replacement API."""
import argparse
import base64
import importlib.util
from pathlib import Path
import json
import re
import sys
from replacement_support import atomic_json,canonical,digest,file_lock,read_json,validate_request


def submit(api,path,state_dir):
    path=Path(path).resolve();raw=read_json(path)
    if not isinstance(raw,dict):raise ValueError('Request must be an object')
    request=dict(raw);files={};fingerprints={}
    for role in ('source','character'):
        key=role+'_path'
        if key not in request:continue
        if role in request:raise ValueError('Supply '+role+' or '+key+', not both')
        value=request.pop(key)
        if not isinstance(value,str):raise ValueError(key+' must be a file path')
        source=Path(value).expanduser()
        if not source.is_absolute():source=path.parent/source
        allowed=('.mp4','.mov','.mkv','.webm','.avi') if role=='source' else ('.png','.jpg','.jpeg','.webp')
        if source.suffix.lower() not in allowed:raise ValueError('Unsupported '+role+' file format')
        if not source.is_file() or not 0<source.stat().st_size<=60*1024**2:
            raise ValueError('Use a regular, nonempty POC input smaller than 60 MB')
        files[role]=source;fingerprints[role]=digest(source);request[role]=source.name
    request=validate_request(request)
    scope=getattr(api,'host_scope','default')
    semantic={key:value for key,value in request.items() if key not in files}
    fingerprint=canonical(dict(request=semantic,files=fingerprints,host=scope))
    state_dir=Path(state_dir);state_dir.mkdir(parents=True,exist_ok=True)
    key=canonical(dict(host=scope,ref=request['ref']))
    record=state_dir/('run-'+key+'.json')
    with file_lock(state_dir/('run-'+key+'.lock')):
        if record.exists():
            saved=read_json(record)
            if saved['fingerprint']!=fingerprint:raise ValueError('This ref already belongs to different inputs')
            if saved.get('id'):return api.request('GET','/replacement/jobs/'+saved['id'])
        doctor=api.request('GET','/replacement')
        if doctor.get('api_version')!='0.2':raise ValueError('Replacement API 0.2 is required; deploy the matching server package')
        if not doctor.get('ready'):
            raise ValueError('Replacement runtime is not ready; run doctor before uploading')
        for role,source in files.items():
            slot=state_dir/('upload-'+canonical(dict(host=scope,role=role,sha256=fingerprints[role]))+'.json')
            with file_lock(slot.with_suffix('.lock')):
                if slot.exists():request[role]=read_json(slot)['file']
                else:
                    data=source.read_bytes()
                    import hashlib
                    if hashlib.sha256(data).hexdigest()!=fingerprints[role]:raise ValueError('Input changed before upload')
                    upload=api.request('POST','/media',dict(name=source.name,data_base64=base64.b64encode(data).decode()))
                    file_ref=upload.get('file') or (upload.get('media') or {}).get('file')
                    if not isinstance(file_ref,str) or not file_ref:raise RuntimeError('Upload returned no file reference')
                    atomic_json(slot,dict(file=file_ref,sha256=fingerprints[role]));request[role]=file_ref
        atomic_json(record,dict(fingerprint=fingerprint,ref=request['ref']))
        result=api.request('POST','/replacement/jobs',request)
        if not re.fullmatch(r'r[0-9a-f]{12}',str(result.get('id',''))):raise RuntimeError('Server returned no replacement job ID')
        atomic_json(record,dict(fingerprint=fingerprint,ref=request['ref'],id=result['id']))
        return result


def main():
    project=Path(__file__).resolve().parent
    default=project.parent/('lora-pipeline' if (project.parent/'lora-pipeline').exists() else 'freevideo-lora')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',default=str(default));parser.add_argument('--config')
    parser.add_argument('--state',default=str(project/'state'))
    sub=parser.add_subparsers(dest='action',required=True);sub.add_parser('doctor')
    run=sub.add_parser('run');run.add_argument('--request',required=True)
    status=sub.add_parser('status');status.add_argument('id');status.add_argument('--wait',type=int,default=0)
    for action in ('cancel','resume'):sub.add_parser(action).add_argument('id')
    args=parser.parse_args();pipeline=Path(args.pipeline)
    spec=importlib.util.spec_from_file_location('freevideo_lora_transport',pipeline/'freevideo_lora.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=module.read_json(args.config or pipeline/'config.json')
    if config.get('transport') not in ('local','ssh'):raise ValueError('Unknown transport configuration')
    api=module.LocalTransport(config) if config['transport']=='local' else module.SSHTransport(config)
    if args.action=='doctor':result=api.request('GET','/replacement')
    elif args.action=='run':result=submit(api,args.request,args.state)
    else:
        if not re.fullmatch(r'r[0-9a-f]{12}',args.id):raise ValueError('Invalid job ID')
        if args.action=='status':result=api.request('GET','/replacement/jobs/'+args.id+'?wait='+str(min(25,max(0,args.wait))))
        else:result=api.request('POST','/replacement/jobs/'+args.id+'/'+args.action,{})
    if isinstance(result,dict) and isinstance(result.get('error'),str):result['error']=result['error'][-1000:]
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(error=str(error))));sys.exit(1)
