"""Bounded-region RGB composition and a lossless, verified video master."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import uuid


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def command(args):
    result = subprocess.run([ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error'] + args,
                            capture_output=True,timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-3000:])
    return result.stdout


def blend(source, generated, source_mask, generated_mask, padding=8, feather=3):
    import numpy as np
    from PIL import Image, ImageFilter
    if source.shape != generated.shape or source.shape[:2] != source_mask.shape or source_mask.shape != generated_mask.shape:
        raise ValueError('Source, generated frame and masks must share geometry')
    union = (source_mask > 0) | (generated_mask > 0)
    if not union.any() or union.mean() > .90:
        raise ValueError('Empty or near-full-frame person mask requires review')
    region = Image.fromarray(union.astype('uint8') * 255)
    region = region.filter(ImageFilter.MaxFilter(2 * padding + 1))
    # Blur only inward: pixels outside the explicit region remain untouched.
    support = np.asarray(region) > 0
    if support.mean() > .95:
        raise ValueError('Edit region covers nearly the whole frame; refine masks or use a wider shot')
    alpha = np.asarray(region.filter(ImageFilter.GaussianBlur(feather)), dtype=np.float32) / 255
    alpha[~support] = 0
    alpha = alpha[..., None]
    result = np.rint(source.astype('float32') * (1-alpha) + generated.astype('float32') * alpha).clip(0,255).astype('uint8')
    if not np.array_equal(result[~support], source[~support]):
        raise AssertionError('Compositor changed protected pixels')
    return result, support


def compose(source_dir, generated_dir, source_masks, generated_masks, original, destination, fps, progress=None):
    """Publish the directory only after master and preview both pass verification."""
    target=Path(destination)
    if target.exists() and any(target.iterdir()):raise ValueError('Output directory already contains files')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.with_name('.'+target.name+'-'+uuid.uuid4().hex+'.partial')
    try:
        report=_compose(source_dir,generated_dir,source_masks,generated_masks,original,staging,fps,progress)
        if target.exists():target.rmdir()
        report.update(master=str(target/'master.mkv'),preview=str(target/'preview.mp4'))
        from replacement_support import atomic_json
        atomic_json(staging/'verification.json',report)
        staging.replace(target)
        return report
    finally:
        if staging.exists():shutil.rmtree(staging)


def sequence(folder):
    paths=sorted(Path(folder).glob('*.png'))
    if not paths or [p.name for p in paths]!=[f'{i:06d}.png' for i in range(len(paths))]:
        raise ValueError('Frame sequence must be contiguous and start at 000000.png')
    if any(p.is_symlink() for p in paths):raise ValueError('Symlinked frames are unsupported')
    return paths


def _compose(source_dir, generated_dir, source_masks, generated_masks, original, destination, fps, progress=None):
    import numpy as np
    from PIL import Image
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files=sequence(source_dir);generated=sequence(generated_dir)
    if any(len(sequence(folder))!=len(files) for folder in (generated_dir,source_masks,generated_masks)):
        raise ValueError('Source, generated video and masks must have identical frame counts')
    with Image.open(files[0]) as first:
        width, height = first.size
    master = destination / 'master.mkv'
    silent = destination / 'silent.mkv'
    preview = destination / 'preview.mp4'
    masks = destination / 'edit-region'
    masks.mkdir(exist_ok=True)
    rate = str(Fraction(fps))
    encoder_log=tempfile.TemporaryFile()
    encoder = subprocess.Popen([ffmpeg(), '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',rate,'-i','pipe:0',
        '-an','-c:v','ffv1','-level','3','-pix_fmt','bgr0',str(silent)], stdin=subprocess.PIPE, stderr=encoder_log)
    try:
        for index, path in enumerate(files):
            source = np.asarray(Image.open(path).convert('RGB'))
            if source.shape!=(height,width,3):raise ValueError('Source frame dimensions changed')
            replacement = np.asarray(Image.open(generated[index]).convert('RGB').resize((width,height),Image.Resampling.LANCZOS))
            def mask(folder):
                return np.asarray(Image.open(Path(folder)/f'{index:06d}.png').convert('L').resize((width,height),Image.Resampling.NEAREST))
            frame, region = blend(source,replacement,mask(source_masks),mask(generated_masks))
            Image.fromarray(region.astype('uint8')*255).save(masks/f'{index:06d}.png')
            encoder.stdin.write(frame.tobytes())
            if progress: progress(index+1,len(files))
        encoder.stdin.close()
        if encoder.wait(timeout=300):
            encoder_log.seek(0);raise RuntimeError(encoder_log.read().decode(errors='replace')[-2000:])
    finally:
        if encoder.poll() is None: encoder.kill(); encoder.wait()
        if not encoder.stdin.closed:
            try:encoder.stdin.close()
            except BrokenPipeError:pass
        encoder_log.close()
    command(['-y','-i',str(silent),'-i',str(original),'-map','0:v:0','-map','1:a?',
             '-c','copy','-t',str(float(Fraction(len(files),1)/Fraction(fps))),str(master)])
    # Verify every protected RGB pixel after real FFV1 encoding and decoding.
    decoder_log=tempfile.TemporaryFile()
    decoder = subprocess.Popen([ffmpeg(),'-nostdin','-hide_banner','-loglevel','error',
        '-i',str(master),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=decoder_log)
    protected = 0
    try:
        for index,path in enumerate(files):
            count=width*height*3
            parts=[];remaining=count
            while remaining:
                part=decoder.stdout.read(remaining)
                if not part:raise RuntimeError('Master lost frames during encoding')
                parts.append(part);remaining-=len(part)
            frame=np.frombuffer(b''.join(parts),dtype=np.uint8).reshape(height,width,3)
            source=np.asarray(Image.open(path).convert('RGB'))
            region=np.asarray(Image.open(masks/f'{index:06d}.png'))>0
            if not np.array_equal(frame[~region],source[~region]):
                raise RuntimeError(f'Master changed protected pixels at frame {index}')
            protected+=int((~region).sum())
        if decoder.stdout.read(1):raise RuntimeError('Master contains extra frames')
        if decoder.wait(timeout=300):
            decoder_log.seek(0);raise RuntimeError(decoder_log.read().decode(errors='replace')[-2000:])
    finally:
        if decoder.poll() is None:decoder.kill();decoder.wait()
        decoder.stdout.close();decoder_log.close()
    command(['-y','-i',str(master),'-map','0:v:0','-map','0:a?',
             '-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','18',
             '-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(preview)])
    import av
    with av.open(original) as source_media,av.open(master) as output:
        stream=output.streams.video[0]
        if Fraction(stream.average_rate)!=Fraction(fps) or (stream.width,stream.height)!=(width,height):
            raise RuntimeError('Master frame rate or dimensions changed')
        audio_codecs=[s.codec_context.name for s in source_media.streams.audio]
        if [s.codec_context.name for s in output.streams.audio]!=audio_codecs:
            raise RuntimeError('Source audio streams were not preserved')
    # Preview decoding catches mux/encoder errors before publication.
    command(['-i',str(preview),'-f','null','-'])
    report=dict(verified=True,frames=len(files),width=width,height=height,fps=rate,
                protected_pixels=protected,maximum_protected_rgb_error=0,
                master=str(master),preview=str(preview),original_audio='source audio streams copied to MKV when present',
                audio_codecs=audio_codecs,preview_decoded=True,
                note='Pixel guarantee applies to lossless master outside saved edit-region masks; preview is lossy.')
    (destination/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    silent.unlink()
    return report
