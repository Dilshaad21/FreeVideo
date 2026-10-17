import io
import hashlib
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(os.environ.get('FREEVIDEO_SOURCE','work/freevideo-source')).resolve()))
from PIL import Image
from freevideo_engine.agent.training import Training, contracts, write_json, read_json, identity, inspect_adapter
from freevideo_engine.agent import schema
from freevideo_engine.agent.training_worker import build_stages

class Port:
    def __init__(self,root):self.root=root
    def input_path(self,relative):
        p=(self.root/'input'/relative).resolve()
        p.relative_to((self.root/'input').resolve())
        return p
    def snapshot(self):return [],[]

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/'input').mkdir()
        self.service=Training(self.root,Port(self.root))
        self.dataset=self.service.create_dataset(dict(name='Example subject',trigger='fv_testperson',subject='person'))
    def tearDown(self):self.tmp.cleanup()
    def photo(self,name,color):
        p=self.root/'input'/name
        Image.new('RGB',(512,640),color).save(p)
        return dict(file=name,angle='front',caption='person looking forward, grey shirt, plain background')
    def test_contracts_validate_real_types_and_unknown_fields(self):
        c=contracts()
        body=c[('POST','/training/jobs')]['body']
        schema.validate(body,dict(dataset=self.dataset['id'],ref='test',rank=8))
        for value in [dict(dataset='../escape',ref='test'),dict(dataset=self.dataset['id'],ref='test',rank='8'),
                      dict(dataset=self.dataset['id'],ref='test',command='calc.exe')]:
            with self.assertRaises(ValueError):schema.validate(body,value)
    def test_identity_cannot_escape(self):
        for value in ['../dataset','d'+'a'*12+'/bad','d'+'a'*12+'\\bad','D'+'a'*12]:
            with self.assertRaises(ValueError):identity(value,'d')
    def test_adapter_validation_rejects_truncated_tensor_payload(self):
        import struct
        path=self.root/'adapter.safetensors'
        header=json.dumps({
            'lora_unet_blocks_0.x.lora_down.weight':dict(shape=[8,16],data_offsets=[0,256]),
            'lora_unet_blocks_0.x.lora_up.weight':dict(shape=[16,8],data_offsets=[256,512])}).encode()
        path.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*512)
        self.assertTrue(inspect_adapter(path)['verified'])
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'Truncated adapter tensors'):inspect_adapter(path)
    def test_upload_caption_dedup_and_coverage(self):
        row=self.service.add_photo(self.dataset['id'],self.photo('a.png','red'))
        self.assertEqual(row['coverage']['front'],1)
        self.assertTrue(row['photos'][0]['caption'].startswith('fv_testperson, '))
        self.assertIn('left-profile',row['suggestions'])
        with self.assertRaisesRegex(ValueError,'already'):self.service.add_photo(self.dataset['id'],self.photo('b.png','red'))
        self.assertEqual(len(self.service.dataset(row['id'])['photos']),1)
    def test_source_hash_upload_is_idempotent_and_detects_changed_caption(self):
        photo=self.photo('replay.png','blue')
        photo['source_sha256']=hashlib.sha256((self.root/'input'/'replay.png').read_bytes()).hexdigest()
        first=self.service.add_photo(self.dataset['id'],photo)
        second=self.service.add_photo(self.dataset['id'],photo)
        self.assertEqual(first['photos'],second['photos'])
        self.assertEqual(len(second['photos']),1)
        with self.assertRaisesRegex(ValueError,'different caption'):
            self.service.add_photo(self.dataset['id'],dict(photo,caption='changed caption'))
        with self.assertRaisesRegex(ValueError,'hash does not match'):
            self.service.add_photo(self.dataset['id'],dict(photo,source_sha256='0'*64))
    def test_caption_changes_are_checked_before_writing(self):
        row=self.service.add_photo(self.dataset['id'],self.photo('a.png','red'))
        with self.assertRaises(ValueError):self.service.captions(row['id'],dict(photos=[dict(id='p'+'0'*12,caption='bad',angle='back')]))
        photo=row['photos'][0]
        changed=self.service.captions(row['id'],dict(photos=[dict(id=photo['id'],caption='left side, blue shirt',angle='left-profile')]))
        self.assertEqual(changed['coverage']['left-profile'],1)
        self.assertIn('blue shirt',changed['photos'][0]['caption'])
    def test_unreadable_file_and_traversal(self):
        (self.root/'input'/'invalid.png').write_text('not an image')
        with self.assertRaises(ValueError):self.service.add_photo(self.dataset['id'],dict(file='invalid.png',angle='front',caption='a face'))
        with self.assertRaises(ValueError):self.service.add_photo(self.dataset['id'],dict(file='../bad',angle='front',caption='a face'))
    def test_trigger_is_unique(self):
        with self.assertRaises(ValueError):self.service.create_dataset(dict(name='Other',trigger='FV_TESTPERSON',subject='person'))
    def test_not_ready_does_not_spawn(self):
        with patch('subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'not ready'):self.service.start(dict(dataset=self.dataset['id'],ref='x'))
            spawn.assert_not_called()
    def test_dead_worker_becomes_interrupted(self):
        row=dict(id='t'+'a'*12,dataset=self.dataset['id'],status='running',pid=2147483647,process_created=1,ref='x')
        write_json(self.service.run_path(row['id']),row)
        self.assertEqual(self.service.run(row['id'])['status'],'interrupted')
    def test_cancel_writes_flag_only_for_live_work(self):
        row=dict(id='t'+'b'*12,dataset=self.dataset['id'],status='waiting',pid=None,ref='x')
        write_json(self.service.run_path(row['id']),row)
        self.service.cancel(row['id'])
        self.assertTrue((self.service.run_path(row['id']).parent/'cancel').exists())
        with self.assertRaises(ValueError):self.service.require_editable(self.dataset['id'])
    def test_failed_stage_never_registers_a_lora(self):
        import os
        from contextlib import nullcontext
        from freevideo_engine.agent import training_worker
        row=self.service.add_photo(self.dataset['id'],self.photo('a.png','red'))
        name='t'+'c'*12
        job=dict(id=name,dataset=row['id'],status='waiting',pid=os.getpid()+1,rank=8,epochs=1,megapixels=.25)
        write_json(self.service.run_path(name),job)
        config=dict(python='train-python',repo=str(self.root),dit='dit',vae='vae',text_encoder='te',training_adapter='adapter',lora_dir=str(self.root/'loras'))
        write_json(self.service.home/'backend.json',config)
        class FailedProcess:
            returncode=1
            def poll(self):return 1
        with patch.object(sys,'argv',['worker','--root',str(self.root),'--job',name]), \
             patch.object(training_worker,'runtime_lock',return_value=nullcontext()), \
             patch.object(training_worker.subprocess,'Popen',return_value=FailedProcess()):
            with self.assertRaisesRegex(RuntimeError,'latents failed'):training_worker.main()
        result=json.loads(self.service.run_path(name).read_text())
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('lora',result)
        self.assertTrue((self.service.run_path(name).parent/'dataset.json').exists())
        self.assertFalse((self.root/'loras').exists())

    def test_status_readers_and_writer_can_run_concurrently(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        target=self.root/'concurrent.json'
        write_json(target,dict(value=0,padding='x'*8000))
        stop=threading.Event()
        def reader():
            while not stop.is_set():
                row=read_json(target)
                self.assertIsInstance(row['value'],int)
                self.assertEqual(len(row['padding']),8000)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(reader) for _ in range(4)]
            try:
                for i in range(50):write_json(target,dict(value=i,padding='x'*8000))
            finally:stop.set()
            for future in futures:future.result()
        self.assertEqual(read_json(target)['value'],49)

    def test_windows_open_reader_does_not_abort_atomic_save(self):
        import os
        original=os.replace
        calls=[]
        def busy_once(a,b):
            calls.append(1)
            if len(calls)==1:raise PermissionError('Windows reader holds the destination briefly')
            return original(a,b)
        target=self.root/'record.json'
        with patch('freevideo_engine.agent.training.os.replace',side_effect=busy_once):
            write_json(target,dict(status='running'))
        self.assertEqual(json.loads(target.read_text())['status'],'running')
        self.assertEqual(len(calls),2)

    def test_setup_log_handles_windows_utf16(self):
        (self.service.home/'setup.log').write_text('Downloading training dependencies',encoding='utf-16')
        self.assertEqual(self.service.setup_log(),'Downloading training dependencies')

    def test_three_stages_use_pinned_cli_and_isolated_interpreter(self):
        config=dict(python='train-python',repo='/trainer',dit='/model/h3.safetensors',vae='/model/vae.safetensors',
                    text_encoder='/model/te.safetensors',training_adapter='/model/adapter.safetensors')
        row=dict(id='t'+'a'*12,dataset=self.dataset['id'],rank=8,epochs=50)
        name,stages=build_stages(config,row,self.root,self.dataset)
        self.assertEqual([x[0] for x in stages],['latents','text','training'])
        self.assertTrue(all(cmd[0]=='train-python' for _,cmd in stages))
        self.assertGreater(int(stages[0][1][stages[0][1].index('--num_workers')+1]),0)
        self.assertIn('--training_adapter',stages[2][1])
        self.assertIn('--blocks_to_swap',stages[2][1])
        self.assertNotIn('--sample_at_first',stages[2][1])

if __name__=='__main__':unittest.main()
