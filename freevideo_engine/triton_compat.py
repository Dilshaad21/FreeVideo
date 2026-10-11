"""Windows Unicode compatibility for compiler configuration and kernel loading."""
from functools import wraps
import os
from pathlib import Path


COMPILER_ENVIRONMENT_KEYS = (
    'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH',
    'CUDA_CACHE_MAXSIZE', 'TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER',
)
# The driver's JIT cache limit; 4 GiB is the most it accepts. RTX 50 kernels
# ship as capsule-Mercury cubins, and a Windows driver (616.92) finalized
# them again on first load and kept the result in CUDA_CACHE_PATH, entries up
# to 58 MiB (10 MiB in all on Linux). At the default 1 GiB the cache filled,
# the driver evicted entries, and finalizing one again stalled encoding, the
# first step, the latent upscale or decoding for about 10 s, at random.
CUDA_CACHE_MAXSIZE = str(4 << 30)


def torch_default_inductor_cache():
    """torch.compile's own default, as torch._inductor.runtime.cache_dir_utils builds it.

    The first time torch needs this folder it writes it back into os.environ.
    A ComfyUI process that has touched torch.compile then hands the temporary
    folder to every child as if a caller had chosen it, and system cleanup of
    temporary files deletes the compiled graphs.
    """
    import getpass
    import re
    import tempfile
    try:
        username = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        getuid = getattr(os, 'getuid', None)
        username = f'uid_{getuid()}' if callable(getuid) else 'unknown_user'
    return os.path.join(tempfile.gettempdir(), 'torchinductor_' + re.sub(r'[\\/:*?"<>|]', '_', username))


def _same_path(first, second):
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def environment(root, environ):
    """Configure child compilers before imports, including encoder prewarming."""
    env = dict(environ)
    cache = Path(root) / 'kernel-cache'
    for key, folder in (('TRITON_CACHE_DIR', 'triton'),
                        ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                        ('CUDA_CACHE_PATH', 'cuda')):
        if not env.get(key):
            env[key] = str(cache / folder)
    # Generation and background precompiling must share one folder, and it
    # belongs to the installation. Only torch's inherited default is replaced;
    # a folder someone set on purpose stays.
    if _same_path(env['TORCHINDUCTOR_CACHE_DIR'], torch_default_inductor_cache()):
        env['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
    if not env.get('CUDA_CACHE_MAXSIZE'):
        env['CUDA_CACHE_MAXSIZE'] = CUDA_CACHE_MAXSIZE
    if os.name == 'nt' and any(not os.path.abspath(env[key]).isascii()
                              for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')):
        # The static launcher passes a UTF-8 narrow filename to cuModuleLoad.
        # Triton's normal launcher loads the same cubin bytes with
        # cuModuleLoadData, avoiding Windows filename encoding at that boundary.
        # Keep explicit caller choices and leave torch.compile enabled.
        env.setdefault('TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER', '0')
    return env


def activate():
    """Recover Unicode knob values when the native getenv binding cannot decode.

    triton-windows 3.7.1.post27 exposes narrow CRT getenv strings through a
    UTF-8 pybind return value. A Chinese TRITON_CACHE_DIR on a GBK Windows
    installation fails before a kernel can compile. Python's Windows
    environment mapping already preserves the original Unicode value.
    """
    if os.name != 'nt':
        return
    from triton import knobs
    native_getenv = knobs.getenv
    if getattr(native_getenv, '_freevideo_unicode_env', False):
        return

    @wraps(native_getenv)
    def unicode_getenv(key, *args, **kwargs):
        try:
            return native_getenv(key, *args, **kwargs)
        except UnicodeDecodeError:
            value = os.environ.get(key)
            if value is None or value.isascii():
                raise
            return value

    unicode_getenv._freevideo_unicode_env = True
    knobs.getenv = unicode_getenv
