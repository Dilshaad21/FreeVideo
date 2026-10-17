"""Read frozen encoder tensors from a read-only file mapping, without a RAM-sized copy."""
import json
import math
import mmap
import struct
import warnings

import torch


class DiskSafeOpen:
    def __init__(self,path):
        self.file=open(path,'rb')
        try:
            size=struct.unpack('<Q',self.file.read(8))[0]
            if size>64*1024**2:raise ValueError('Oversized safetensors header')
            self.header=json.loads(self.file.read(size))
            self.start=8+size
            self.mapping=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):return self

    def __exit__(self,*args):
        # Returned tensors retain their buffer's mapping; closing this file never invalidates them.
        self.file.close()
        self.mapping=None

    def keys(self):return [k for k in self.header if k!='__metadata__']

    def metadata(self):return self.header.get('__metadata__',{})

    def get_tensor(self,name):
        entry=self.header[name]
        types={'F64':torch.float64,'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,
               'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,
               'U8':torch.uint8,'BOOL':torch.bool,'F8_E4M3':torch.float8_e4m3fn,'F8_E5M2':torch.float8_e5m2}
        dtype=types[entry['dtype']]
        a,b=entry['data_offsets']
        count=math.prod(entry['shape'])
        if a<0 or b<a or self.start+b>len(self.mapping):raise ValueError('Invalid tensor offsets')
        if count*torch.empty((),dtype=dtype).element_size()!=b-a:raise ValueError('Invalid tensor length')
        if count==0:return torch.empty(entry['shape'],dtype=dtype)
        # These are frozen source weights: consumers copy to GPU or make dtype conversions, never write in place.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The given buffer is not writable')
            raw=torch.frombuffer(self.mapping,dtype=torch.uint8,count=b-a,offset=self.start+a)
        return raw.view(dtype).reshape(entry['shape'])
