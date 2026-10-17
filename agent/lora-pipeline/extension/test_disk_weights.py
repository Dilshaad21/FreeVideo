import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch
from safetensors.torch import save_file
from safetensors import safe_open
module=importlib.util.spec_from_file_location('disk_weights',sys.argv[1])
m=importlib.util.module_from_spec(module);module.loader.exec_module(m)
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'test.safetensors'
 expected={'packed':torch.randint(0,256,(64,64),dtype=torch.uint8),
           'scale':torch.randn(128).to(torch.bfloat16),'float':torch.randn(24,16),
           'empty':torch.empty(0,8),'fp8':torch.randn(32,32).to(torch.float8_e4m3fn)}
 save_file(expected,str(path))
 with m.DiskSafeOpen(path) as mapped,safe_open(path,framework='pt') as normal:
  got={k:mapped.get_tensor(k) for k in mapped.keys()}
  for k,value in got.items():
   assert value.dtype==normal.get_tensor(k).dtype
   assert torch.equal(value.view(torch.uint8),normal.get_tensor(k).view(torch.uint8)), k
 gc.collect()
 for k,value in got.items():assert torch.equal(value.view(torch.uint8),expected[k].view(torch.uint8)), k
 del got,value
 gc.collect()
print(json.dumps({'disk_mapping_values_exact':True,'mapping_survives_reader_exit':True,'dtypes_tested':5}))
