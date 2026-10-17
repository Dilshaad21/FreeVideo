"""Separate-process SAM tracking and WanGP generation adapter."""
import argparse
import json
import os
from pathlib import Path
import sys


def prepare(config,request):
    import av
    from fractions import Fraction
    from PIL import Image
    from composite import command
    folder=Path(request['folder'])
    import shutil
    if request.get('character'):
        from PIL import ImageOps
        with Image.open(request['character']) as image:
            if image.width*image.height>40_000_000:raise ValueError('Character reference exceeds 40 megapixels')
            image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((2048,2048))
            image.save(folder/'reference.png')
    originals=folder/'originals';frames=folder/'source-frames'
    originals.mkdir(exist_ok=True);frames.mkdir(exist_ok=True)
    with av.open(request['source']) as container:
        stream=container.streams.video[0]
        rate=stream.average_rate
        if not rate or not 1<=float(rate)<=60:raise ValueError('Use a constant frame rate from 1 to 60 fps')
        if max(float(stream.width)/max(stream.height,1),float(stream.height)/max(stream.width,1))>4:
            raise ValueError('Source aspect ratio exceeds the supported 4:1 limit')
        estimate=int(stream.width*stream.height*float(rate)*5*24)
        if shutil.disk_usage(folder).free<max(8*1024**3,estimate):
            raise RuntimeError('resource_limit: insufficient disk for decoded frames and verified master')
        previous=None;count=0;geometry=None
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('Missing frame timestamp')
            timestamp=frame.pts*stream.time_base
            if count==0:
                # Avoid silently shifting source audio relative to its video.
                start=Fraction(container.start_time or 0,1_000_000)
                if abs(float(timestamp-start))>.002:
                    raise ValueError('Normalize video/audio start timestamps before replacement')
            if any(side.type.name=='DISPLAYMATRIX' for side in frame.side_data) or stream.metadata.get('rotate','0') not in ('0',''):
                raise ValueError('Normalize source rotation before replacement')
            if previous is not None and abs(float(timestamp-previous-1/rate))>max(.001,float(1/rate)*.03):
                raise ValueError('Variable frame rate needs normalization before replacement')
            previous=timestamp
            image=frame.to_image().convert('RGB')
            width,height=image.size
            if geometry is not None and geometry!=(width,height):raise ValueError('Changing video dimensions are unsupported')
            geometry=(width,height)
            if min(width,height)<64:raise ValueError('Use a source at least 64 pixels on each side')
            if width*height>3840*2160:raise ValueError('Source exceeds 4K working limit')
            image.save(originals/f'{count:06d}.png')
            workheight=min(480,height)//16*16
            workwidth=max(32,round(width*workheight/height/16)*16)
            image.resize((workwidth,workheight),Image.Resampling.LANCZOS).save(frames/f'{count:06d}.jpg',quality=95)
            count+=1
            if count/float(rate)>5.05:raise ValueError('POC supports one shot of at most 5 seconds')
    if count/float(rate)<1:raise ValueError('Source must be at least one second')
    # Wan uses 4n+1 frames; pad only the conditioning tail, then trim to source count.
    padded=max(5,((count-1+3)//4)*4+1)
    for index in range(count,padded):
        import shutil
        shutil.copyfile(frames/f'{count-1:06d}.jpg',frames/f'{index:06d}.jpg')
    video=folder/'control.mkv'
    command(['-y','-framerate',str(rate),'-i',str(frames/'%06d.jpg'),'-c:v','ffv1',str(video)])
    meta=dict(frames=count,padded_frames=padded,fps=str(rate),width=width,height=height,
        resolution=f'{workwidth}x{workheight}',source=request['source'])
    (folder/'media.json').write_text(json.dumps(meta,indent=2))


def extract(request):
    import av
    from PIL import Image
    target=Path(request['destination']);target.mkdir(exist_ok=True)
    pngs=Path(request['png']);pngs.mkdir(exist_ok=True)
    with av.open(request['video']) as container:
        count=0
        for frame in container.decode(video=0):
            if count>=request['frames']:break
            image=frame.to_image().convert('RGB')
            image.save(pngs/f'{count:06d}.png')
            image.save(target/f'{count:06d}.jpg',quality=95)
            count+=1
    if count<request['frames']:raise RuntimeError('Generated video is shorter than the source')


def trim_masks(request):
    import shutil
    source=Path(request['source']);target=Path(request['destination']);target.mkdir(exist_ok=True)
    for index in range(request['frames']):
        path=source/f'{index:06d}.png'
        if not path.is_file() or path.is_symlink():raise ValueError('Missing or invalid source mask')
        shutil.copyfile(path,target/path.name)


def session(home):
    home=Path(home)
    sys.path.insert(0,str(home/'WanGP'))
    os.chdir(home/'WanGP')
    from shared.api import init
    api=init(root=home/'WanGP',output_dir=home/'generated',
             cli_args=['--attention','sdpa','--profile','5'],console_output=True)
    runtime=api._ensure_runtime().module
    runtime.server_config.update(keep_intermediate_sliding_windows=0,enhancer_enabled=0)
    runtime.transformer_dtype_policy='bf16'
    return api


def track(config,frames,destination,box):
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0,config['sam2'])
    from sam2.build_sam import build_sam2_video_predictor
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    predictor=build_sam2_video_predictor(config['sam_config'],config['sam_checkpoint'],device='cuda',
        apply_postprocessing=False)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
        with Image.open(Path(frames)/'000000.jpg') as first:
            width,height=first.size
        bbox=np.array(box,dtype=np.float32)*np.array([width,height,width,height])
        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,box=bbox)
        count=0
        for index,objects,logits in predictor.propagate_in_video(state):
            mask=(logits[0,0]>0).cpu().numpy().astype('uint8')*255
            if not mask.any() or (mask>0).mean()>.90:raise ValueError(f'Tracking mask needs review at frame {index}')
            Image.fromarray(mask).save(destination/f'{index:06d}.png');count+=1
        if count!=state['num_frames']:raise RuntimeError('Tracker did not cover all frames')


def generate(config,request):
    api=session(config['home'])
    settings=api.get_default_settings('animate')
    settings.update(model_type='animate',prompt=request['prompt'],negative_prompt='distorted face, flickering, extra limbs',
        resolution=request['resolution'],num_inference_steps=config.get('steps',20),
        video_length=request['frames'],force_fps='control',seed=42,
        video_prompt_type='PVBAIH#1',image_prompt_type='',
        image_refs=[request['character']],video_guide=request['video'],video_mask=request['mask'],
        remove_background_images_ref=0,mask_expand=0,
        sliding_window_size=81,sliding_window_overlap=5,prompt_enhancer='',audio_prompt_type='')
    job=api.submit_task(settings)
    for event in job.events.iter(timeout=.5):
        if event.kind=='progress':
            value=event.data
            print(json.dumps(dict(phase=getattr(value,'phase',None),step=getattr(value,'current_step',None),
                                  total=getattr(value,'total_steps',None))),flush=True)
    result=job.result()
    if not result.success:raise RuntimeError('; '.join(e.message for e in result.errors))
    videos=[str(p) for p in result.generated_files if str(p).lower().endswith(('.mp4','.mkv','.mov'))]
    if len(videos)!=1:raise RuntimeError('Expected exactly one generated video, received '+repr(videos))
    from replacement_support import atomic_json,digest
    video=Path(videos[0])
    if video.is_symlink():raise ValueError('Symlinked generated video is unsupported')
    video=video.resolve()
    if not video.is_relative_to((Path(config['home'])/'generated').resolve()):
        raise ValueError('Generation returned a file outside its managed output directory')
    atomic_json(request['result'],dict(video=str(video),sha256=digest(video),phase_key=request.get('_phase_key')))


def models(home):
    # Use pinned WanGP's own asset resolver, which handles its tokenizer/VAE layout.
    api=session(home)
    runtime=api._ensure_runtime()
    module=runtime.module
    from shared.utils import download as downloads
    from prefetch import install_wangp_downloader
    install_wangp_downloader(downloads,Path(home)/'asset-progress.json')
    model=module.get_model_def('animate')
    main='https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/wan2.2_animate_14B_quanto_bf16_int8.safetensors'
    from setup_windows import download
    location=Path(home)/'WanGP/ckpts'
    download(main,location/'wan2.2_animate_14B_quanto_bf16_int8.safetensors')
    module.download_models(main,model_type='animate',model_def=model)
    # Text encoder is a separate optional weight in WanGP's loader.
    download('https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/models_t5_umt5-xxl-enc-quanto_int8.safetensors',
             location/'models_t5_umt5-xxl-enc-quanto_int8.safetensors')
    availability=api.get_model_availability('animate')
    Path(home,'model-availability.json').write_text(json.dumps(availability,indent=2))
    if not availability.get('available'):
        raise RuntimeError('WanGP does not recognize the staged Animate checkpoint; inspect model-availability.json')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
        str(Path(home)/'models/sam2.1_hiera_tiny.pt'),device='cpu',apply_postprocessing=False)
    del predictor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['probe','models','prepare','track','generate','extract','trim-masks','mask-video','composite'])
    parser.add_argument('--home',required=True)
    parser.add_argument('--request')
    args=parser.parse_args()
    home=Path(args.home).resolve()
    if args.action=='probe':
        import torch,cv2,av
        from sam2.build_sam import build_sam2_video_predictor
        if not torch.cuda.is_available():raise RuntimeError('Replacement requires the Windows NVIDIA GPU')
        api=session(home)
        # Check the selected generation implementation, not only UI/schema imports.
        from models.wan import any2video as wan_backend
        defaults=api.get_default_settings('animate')
        checkpoint=home/'models/sam2.1_hiera_tiny.pt'
        if checkpoint.is_file():
            predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),
                                               device='cpu',apply_postprocessing=False)
            del predictor
        (home/'animate-defaults.json').write_text(json.dumps(defaults,indent=2,default=str))
        print(json.dumps(dict(runtime_imports=True,cuda=torch.cuda.is_available(),backend='WanGP / Wan2.2-Animate')))
    elif args.action=='models':models(home)
    else:
        config=json.loads((home/'backend.json').read_text())
        request=json.loads(Path(args.request).read_text())
        phase_key=request.pop('_phase_key',None)
        if args.action=='track':track(config,request['frames'],request['destination'],request['box'])
        elif args.action=='generate':generate(config,dict(request,_phase_key=phase_key))
        elif args.action=='prepare':prepare(config,request)
        elif args.action=='extract':extract(request)
        elif args.action=='trim-masks':trim_masks(request)
        elif args.action=='mask-video':
            from composite import command
            command(['-y','-framerate',request['fps'],'-i',str(Path(request['masks'])/'%06d.png'),
                     '-c:v','ffv1',request['output']])
        else:
            from composite import compose
            compose(**request)


if __name__=='__main__':main()
