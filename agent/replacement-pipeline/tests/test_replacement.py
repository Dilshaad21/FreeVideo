import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import composite
import install_extension


def service_module():
    package=types.ModuleType('poc_agent');package.__path__=[str(PROJECT)]
    sys.modules['poc_agent']=package
    training=types.ModuleType('poc_agent.training')
    training.read_json=lambda path:json.loads(Path(path).read_text())
    def write(path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
    training.write_json=write
    def identity(value,prefix):
        import re
        if not re.fullmatch(prefix+'[0-9a-f]{12}',value):raise ValueError('Invalid ID')
        return value
    training.identity=identity
    sys.modules['poc_agent.training']=training
    spec=importlib.util.spec_from_file_location('poc_agent.replacement',PROJECT/'replacement.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.module=service_module()

    def test_box_validation(self):
        base=dict(ref='poc',source='source.mp4',character='person.png')
        for box in ([0,0,0,1],[0,0,2,1],[0,1,1,0],[0,0,float('nan'),1],[True,0,1,1],[]):
            with self.assertRaises(ValueError):self.module.validate(dict(base,box=box))
        self.assertEqual(self.module.validate(dict(base,box=[0,0,1,1]))['box'],[0,0,1,1])

    def test_not_ready_until_runtime_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.module.Replacement(tmp,None)
            self.assertFalse(service.status()['ready'])
            self.assertIn('sam_checkpoint',service.status()['missing'])

    def test_replay_does_not_spawn(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);media=root/'input';media.mkdir()
            (media/'source.mp4').write_bytes(b'source');(media/'person.png').write_bytes(b'person')
            port=types.SimpleNamespace(input_path=lambda name:media/name)
            service=self.module.Replacement(root,port)
            value=self.module.validate(dict(ref='poc',source='source.mp4',character='person.png',box=[.1,.1,.9,.9]))
            semantic={key:item for key,item in value.items() if key not in ('source','character')}
            fingerprint=hashlib.sha256(json.dumps(dict(semantic,video_hash=hashlib.sha256(b'source').hexdigest(),
                character_hash=hashlib.sha256(b'person').hexdigest()),sort_keys=True).encode()).hexdigest()
            row=dict(value,id='r000000000001',status='succeeded',fingerprint=fingerprint)
            self.module.write_json(service.home/'runs'/row['id']/'job.json',row)
            self.assertEqual(service.start(value),row)
            (media/'reuploaded.mp4').write_bytes(b'source')
            (media/'reuploaded.png').write_bytes(b'person')
            self.assertEqual(service.start(dict(value,source='reuploaded.mp4',character='reuploaded.png')),row)
            with self.assertRaises(ValueError):service.start(dict(value,box=[.2,.1,.9,.9]))

    def test_installer_idempotent_and_unknown_refused(self):
        text='def register():\n    contracts.update(training.contracts())\n    server._freevideo_training = training.register(routes, guarded, reply, body, port, jobs)\n'
        installed=install_extension.prepare(text)
        self.assertEqual(install_extension.prepare(installed),installed)
        self.assertEqual(installed.count('replacement.register('),1)
        with self.assertRaises(RuntimeError):install_extension.prepare('unknown layout')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'module.py';data='snowman=\"☃\"\n'.encode('utf-8')
            install_extension.publish(path,data);self.assertEqual(path.read_bytes(),data)


class CompositorTests(unittest.TestCase):
    def test_prepare_keeps_source_frames_and_pads_conditioning_only(self):
        from PIL import Image
        import runtime
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);inputs=folder/'input';inputs.mkdir()
            for index in range(24):Image.new('RGB',(96,64),(index*10,30,80)).save(inputs/f'{index:06d}.png')
            original=folder/'source.mkv'
            composite.command(['-y','-framerate','24','-i',str(inputs/'%06d.png'),'-c:v','ffv1',str(original)])
            target=folder/'job';target.mkdir()
            runtime.prepare({},dict(folder=str(target),source=str(original)))
            meta=json.loads((target/'media.json').read_text())
            self.assertEqual(meta['frames'],24);self.assertEqual(meta['padded_frames'],25)
            from fractions import Fraction
            self.assertEqual(Fraction(meta['fps']),Fraction(24,1))
            self.assertEqual(len(list((target/'originals').glob('*.png'))),24)
            self.assertEqual(len(list((target/'source-frames').glob('*.jpg'))),25)
            self.assertEqual(Image.open(target/'originals/000005.png').getpixel((0,0)),(50,30,80))

    def test_protected_pixels_and_union(self):
        import numpy as np
        source=np.full((64,96,3),37,dtype=np.uint8)
        generated=np.full_like(source,220)
        first=np.zeros((64,96),dtype=np.uint8);first[15:45,30:40]=255
        second=np.zeros_like(first);second[15:45,35:50]=255
        result,region=composite.blend(source,generated,first,second,padding=2,feather=1)
        self.assertTrue(np.array_equal(result[~region],source[~region]))
        self.assertTrue((result[30,45]>source[30,45]).all())

    def test_empty_and_full_masks_refused(self):
        import numpy as np
        frame=np.zeros((64,64,3),dtype=np.uint8)
        for mask in (np.zeros((64,64),dtype=np.uint8),np.full((64,64),255,dtype=np.uint8)):
            with self.assertRaises(ValueError):composite.blend(frame,frame,mask,mask)

    def test_real_lossless_roundtrip(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for part in ('source','generated','masks'): (folder/part).mkdir()
            for index in range(4):
                source=np.random.default_rng(index).integers(0,256,(64,96,3),dtype=np.uint8)
                generated=source.copy();generated[15:45,30:50]=[index*50,80,200]
                mask=np.zeros((64,96),dtype=np.uint8);mask[15:45,30:50]=255
                for part,data in [('source',source),('generated',generated),('masks',mask)]:
                    Image.fromarray(data).save(folder/part/f'{index:06d}.png')
            original=folder/'original.mkv'
            composite.command(['-y','-framerate','24','-i',str(folder/'source/%06d.png'),'-c:v','ffv1',str(original)])
            report=composite.compose(folder/'source',folder/'generated',folder/'masks',folder/'masks',original,folder/'output','24')
            self.assertTrue(report['verified']);self.assertEqual(report['frames'],4)
            self.assertEqual(report['maximum_protected_rgb_error'],0)
            self.assertGreater(report['protected_pixels'],0)


if __name__=='__main__':unittest.main()
