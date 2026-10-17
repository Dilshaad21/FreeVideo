import json
from pathlib import Path
import sys
root=Path(sys.argv[1]);home=root/'agent'/'training'
sys.path.insert(0,str(home/'Fizgig'/'src'))
import torch
from fizgig.minimax import common
from fizgig.minimax.driver import MiniMaxDriver
from fizgig.families.registry import get
assert torch.cuda.is_available(), 'CUDA is unavailable in the training environment'
x=torch.ones((32,32),device='cuda')
assert (x@x).sum().item()==32768
result=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,h3_driver=True,
            low_vram_plan=common.plan_vram(7.5,mp=.25,resident_gb=10.5,adapter_gb=.3))
(home/'cuda-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
