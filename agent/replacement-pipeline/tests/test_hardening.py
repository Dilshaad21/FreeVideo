import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import replacement_support as support
import prefetch
import freevideo_replace as client
import composite


class RecoveryTests(unittest.TestCase):
    def test_journal_reuses_only_identical_artifacts_and_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.write_bytes(b'a');b.write_bytes(b'b')
            journal=support.Journal(root,'v1');key=journal.key('first',{});journal.complete('first',key,[a])
            second=journal.key('second',{});journal.complete('second',second,[b])
            replay=support.Journal(root,'v1');self.assertTrue(replay.reusable('first',key,[a]))
            self.assertEqual(replay.key('second',{}),second)
            a.write_bytes(b'x');broken=support.Journal(root,'v1')
            self.assertFalse(broken.reusable('first',key,[a]));broken.complete('first',key,[a])
            self.assertNotEqual(broken.key('second',{}),second)
            b.unlink();broken.verify_all() # stale downstream receipts do not block recomputation
            a.write_bytes(b'y')
            with self.assertRaises(ValueError):broken.verify_all()

    def test_resume_signature_tracks_worker_and_core_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('runtime.py','composite.py','replacement_support.py','replacement_worker.py'):(root/name).write_text('v1')
            weights=root/'WanGP/ckpts';weights.mkdir(parents=True)
            for name in ('wan2.2_animate_14B_quanto_bf16_int8.safetensors','models_t5_umt5-xxl-enc-quanto_int8.safetensors'):(weights/name).write_bytes(b'weight')
            checkpoint=root/'sam.pt';checkpoint.write_bytes(b'sam')
            config=dict(source=str(root),home=str(root),wangp=str(root/'WanGP'),sam2=str(root/'sam2'),sam_checkpoint=str(checkpoint))
            before=support.execution_signature(config);(root/'replacement_worker.py').write_text('v2')
            changed=support.execution_signature(config);self.assertNotEqual(before,changed)
            checkpoint.write_bytes(b'changed');self.assertNotEqual(changed,support.execution_signature(config))
            before=support.execution_signature(config);(root/'dependencies.txt').write_text('torch==changed')
            self.assertNotEqual(before,support.execution_signature(config))

    def test_extra_file_invalidates_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frames=root/'frames';frames.mkdir();(frames/'a').write_text('x')
            journal=support.Journal(root,'v1');journal.complete('frames','key',[frames]);(frames/'extra').write_text('x')
            self.assertFalse(support.Journal(root,'v1').reusable('frames','key',[frames]))

    def test_malformed_receipt_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);journal=support.Journal(root,'v1');(journal.folder/'stage.json').write_text('{')
            self.assertFalse(journal.reusable('stage','key',[root/'absent']))

    def test_process_lock_prevents_lost_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state.json';support.atomic_json(state,dict(counter=0))
            code="import sys;sys.path.insert(0,sys.argv[1]);from replacement_support import mutate\nfor _ in range(20):mutate(sys.argv[2],lambda row:row.update(counter=row['counter']+1))"
            children=[subprocess.Popen([sys.executable,'-c',code,str(PROJECT),str(state)]) for _ in range(3)]
            for child in children:self.assertEqual(child.wait(timeout=30),0)
            self.assertEqual(support.read_json(state)['counter'],60)

    def test_stale_pid_never_counts_as_owned(self):
        import psutil
        self.assertTrue(support.alive(os.getpid(),psutil.Process().create_time()))
        self.assertFalse(support.alive(os.getpid(),psutil.Process().create_time()-5))

    @unittest.skipUnless(os.name=='nt','Windows Job Object integration')
    def test_windows_worker_crash_kills_descendants(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            record=Path(tmp)/'child.json'
            code="import sys,time,subprocess,os;sys.path.insert(0,sys.argv[1]);from replacement_support import bind_worker_tree,atomic_json;import psutil;handle=bind_worker_tree();child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);atomic_json(sys.argv[2],dict(pid=child.pid,created=psutil.Process(child.pid).create_time()));time.sleep(120)"
            worker=subprocess.Popen([sys.executable,'-c',code,str(PROJECT),str(record)])
            child=None
            try:
                deadline=time.monotonic()+15
                while not record.exists() and time.monotonic()<deadline:
                    if worker.poll() is not None:self.fail('Windows job binding failed')
                    time.sleep(.1)
                child=support.read_json(record);worker.kill();worker.wait(timeout=10)
                deadline=time.monotonic()+10
                while support.alive(child['pid'],child['created']) and time.monotonic()<deadline:time.sleep(.1)
                self.assertFalse(support.alive(child['pid'],child['created']))
            finally:
                if worker.poll() is None:worker.kill();worker.wait()
                if child:support.terminate_tree(child['pid'],child['created'])


class ServiceRecoveryTests(unittest.TestCase):
    def setUp(self):
        from test_replacement import service_module
        self.module=service_module()

    def test_dead_worker_becomes_interrupted_and_orphan_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.module.Replacement(tmp,None);name='r000000000001'
            path=service.home/'runs'/name/'job.json'
            support.atomic_json(path,dict(id=name,status='running',created_at=0,pid=100,process_created=1,child_pid=200,child_created=2))
            with patch.object(self.module,'alive',return_value=False):self.assertEqual(service.job(name)['status'],'interrupted')
            row=support.read_json(path);row['status']='running';support.atomic_json(path,row)
            with patch.object(self.module,'alive',side_effect=lambda pid,created:pid==200):
                self.assertEqual(service.job(name)['status'],'orphaned')
                with self.assertRaisesRegex(ValueError,'orphaned'):service.resume(name)

    def test_explicit_resume_preserves_id_and_rejects_changed_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=self.module.Replacement(tmp,None);name='r000000000001'
            path=service.home/'runs'/name/'job.json'
            support.atomic_json(path,dict(id=name,status='failed',created_at=0,execution_signature='same',attempt=1))
            support.atomic_json(service.home/'backend.json',{})
            with patch.object(service,'status',return_value=dict(ready=True)),patch.object(self.module,'execution_signature',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'changed'):service.resume(name)
            (path.parent/'cancel').touch()
            with patch.object(service,'status',return_value=dict(ready=True)),patch.object(self.module,'execution_signature',return_value='same'),patch.object(service,'launch',side_effect=lambda folder:support.read_json(folder/'job.json')):
                row=service.resume(name)
                self.assertEqual((row['id'],row['status'],row['attempt']),(name,'waiting',2))
                self.assertFalse((path.parent/'cancel').exists())


class ResourceTests(unittest.TestCase):
    def test_sustained_heat_stops_execution(self):
        monitor=support.ResourceMonitor({})
        with patch('replacement_support.time.monotonic',side_effect=[0,31]),patch('subprocess.check_output',return_value='90,7000,100'):
            monitor.sample(force=True)
            with self.assertRaisesRegex(RuntimeError,'temperature'):monitor.sample(force=True)

    def test_sustained_low_ram_stops_execution(self):
        from types import SimpleNamespace
        monitor=support.ResourceMonitor({})
        with patch('replacement_support.time.monotonic',side_effect=[0,16]),patch('psutil.virtual_memory',return_value=SimpleNamespace(available=100)),patch('subprocess.check_output',return_value='50,100,0'):
            monitor.sample(force=True)
            with self.assertRaisesRegex(RuntimeError,'RAM'):monitor.sample(force=True)

    def test_missing_gpu_sensor_reported(self):
        with patch('subprocess.check_output',side_effect=FileNotFoundError):
            self.assertFalse(support.ResourceMonitor({}).sample(force=True)['gpu_sensor_available'])


class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {'Content-Length':str(len(data))}


class RuntimeDispatchTests(unittest.TestCase):
    def test_padding_masks_trimmed_to_exact_source_length(self):
        import runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            for i in range(25):(source/f'{i:06d}.png').write_bytes(bytes([i]))
            runtime.trim_masks(dict(source=str(source),destination=str(root/'trimmed'),frames=24))
            self.assertEqual(len(composite.sequence(root/'trimmed')),24)
            self.assertEqual(len(composite.sequence(source)),25)

    def test_models_subcommand_calls_validator(self):
        import runtime
        with tempfile.TemporaryDirectory() as tmp,patch.object(sys,'argv',['runtime.py','models','--home',tmp]),patch.object(runtime,'models') as validate:
            runtime.main();validate.assert_called_once_with(Path(tmp).resolve())


class DownloadTests(unittest.TestCase):
    def fetch(self,root,response):
        with patch('urllib.request.urlopen',return_value=response):prefetch.fetch('https://example.test/asset',root/'asset.bin',root/'progress.json')

    def test_partial_resume_and_checksum_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'asset.bin.part').write_bytes(b'abc')
            self.fetch(root,Response(b'def',206,{'Content-Range':'bytes 3-5/6'}))
            self.assertEqual((root/'asset.bin').read_bytes(),b'abcdef')
            (root/'asset.bin').write_bytes(b'xxxxxx')
            with self.assertRaisesRegex(ValueError,'checksum'):prefetch.fetch('https://example.test/asset',root/'asset.bin',root/'progress.json')

    def test_wrong_range_refused_without_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);partial=root/'asset.bin.part';partial.write_bytes(b'abc')
            with self.assertRaisesRegex(ValueError,'range'):self.fetch(root,Response(b'def',206,{'Content-Range':'bytes 0-2/3'}))
            self.assertEqual(partial.read_bytes(),b'abc');self.assertFalse((root/'asset.bin').exists())

    def test_ignored_range_restarts_and_incomplete_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'asset.bin.part').write_bytes(b'old')
            self.fetch(root,Response(b'new'));self.assertEqual((root/'asset.bin').read_bytes(),b'new')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Incomplete'):self.fetch(root,Response(b'abc',headers={'Content-Length':'6'}))
            self.assertFalse((root/'asset.bin').exists());self.assertEqual((root/'asset.bin.part').read_bytes(),b'abc')

    def test_wangp_asset_adapter_preserves_paths_and_blocks_escape(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            module=SimpleNamespace();root=Path(tmp).resolve();prefetch.install_wangp_downloader(module,root/'state.json')
            with patch.object(prefetch,'fetch') as fetch:
                result=module._hf_download(repo_id='org/model',filename='config.json',subfolder='tokenizer',local_dir=str(root))
                self.assertEqual(result,str(root/'tokenizer/config.json'))
                self.assertEqual(fetch.call_args.args[0],'https://huggingface.co/org/model/resolve/main/tokenizer/config.json')
                with self.assertRaisesRegex(ValueError,'Unsafe'):module._hf_download(repo_id='org/model',filename='../escape',local_dir=str(root))
                with self.assertRaises(ValueError):module._hf_download(repo_id='org/model',filename='a',local_dir=str(root),revision='unrecognized')

    def test_truncated_safetensors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.safetensors';header=json.dumps({'x':{'data_offsets':[0,4]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+b'xx')
            with self.assertRaisesRegex(ValueError,'payload'):prefetch.validate_artifact(path)
            path.write_bytes(struct.pack('<Q',len(header))+header+b'xxxx');prefetch.validate_artifact(path)


class FakeAPI:
    host_scope='windows'
    def __init__(self):self.uploads=0;self.posts=0;self.ready=True;self.lose=False
    def request(self,method,path,value=None):
        if path=='/replacement':return dict(ready=self.ready,api_version='0.2')
        if path=='/media':self.uploads+=1;return dict(file=f'upload{self.uploads}.png')
        if path=='/replacement/jobs':
            self.posts+=1
            if self.lose:self.lose=False;raise TimeoutError('response lost')
            return dict(id='r000000000001',status='waiting')
        return dict(id='r000000000001',status='running')


class ClientTests(unittest.TestCase):
    def request(self,root):
        (root/'source.mp4').write_bytes(b'video');(root/'person.png').write_bytes(b'photo')
        path=root/'request.json';support.atomic_json(path,dict(ref='one',source_path='source.mp4',character_path='person.png',box=[.1,.1,.9,.9]))
        return path

    def test_replays_same_job_without_reupload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.request(root);api=FakeAPI()
            client.submit(api,path,root/'state');result=client.submit(api,path,root/'state')
            self.assertEqual((api.uploads,api.posts),(2,1));self.assertEqual(result['status'],'running')
            (root/'person.png').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'different inputs'):client.submit(api,path,root/'state')

    def test_lost_post_response_reuses_uploads_and_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.request(root);api=FakeAPI();api.lose=True
            with self.assertRaises(TimeoutError):client.submit(api,path,root/'state')
            client.submit(api,path,root/'state');self.assertEqual((api.uploads,api.posts),(2,2))

    def test_not_ready_never_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.request(root);api=FakeAPI();api.ready=False
            with self.assertRaises(ValueError):client.submit(api,path,root/'state')
            self.assertEqual(api.uploads,0)


class PublicationTests(unittest.TestCase):
    def test_missing_frames_never_publish_partial_output(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('originals','generated','mask'):
                folder=root/name;folder.mkdir();Image.new('RGB',(96,64)).save(folder/'000000.png')
            Image.new('RGB',(96,64)).save(root/'originals/000001.png')
            with self.assertRaisesRegex(ValueError,'frame counts'):
                composite.compose(root/'originals',root/'generated',root/'mask',root/'mask',root/'video',root/'output','24')
            self.assertFalse((root/'output').exists());self.assertEqual(list(root.glob('.*.partial')),[])

    def test_audio_preserved_and_odd_size_preview_decodes(self):
        import av
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('originals','generated','mask'):
                folder=root/name;folder.mkdir()
                for i in range(4):
                    image=Image.new('L',(97,65),0) if name=='mask' else Image.new('RGB',(97,65),(40,70,80))
                    if name=='mask':
                        from PIL import ImageDraw
                        ImageDraw.Draw(image).rectangle((30,15,50,45),fill=255)
                    image.save(folder/f'{i:06d}.png')
            original=root/'original.mkv'
            composite.command(['-y','-framerate','24','-i',str(root/'originals/%06d.png'),'-f','lavfi','-i','sine=frequency=440:duration=0.167','-c:v','ffv1','-c:a','flac','-shortest',str(original)])
            report=composite.compose(root/'originals',root/'generated',root/'mask',root/'mask',original,root/'output','24')
            self.assertEqual(report['audio_codecs'],['flac']);self.assertTrue(report['preview_decoded'])
            with av.open(root/'output/preview.mp4') as media:self.assertEqual((media.streams.video[0].width,media.streams.video[0].height),(98,66))
            self.assertEqual(support.read_json(root/'output/verification.json')['master'],str(root/'output/master.mkv'))


if __name__=='__main__':unittest.main()
