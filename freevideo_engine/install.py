"""Install model code at a pinned commit without changing the user's models."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from . import execution_paths
from .paths import vdn_root
from . import network


def install(root=None, *, packages=False):
    dependencies = json.loads((Path(__file__).resolve().parent / 'dependencies.json').read_text(encoding='utf-8'))
    spec = dependencies['vdn']
    root = (root or vdn_root()).expanduser().resolve()
    network.clone(root, spec['url'], spec['commit'])
    head = execution_paths.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = execution_paths.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True)
    if head != spec['commit'] or dirty:
        raise RuntimeError('Existing VDN checkout does not match the clean pinned dependency; use another --vdn-root.')
    diffusers = prepare_diffusers(root, dependencies)
    if packages:
        # Install Torch explicitly first; preserve that CUDA build here.
        if not os.environ.get('FREEVIDEO_UV') and importlib.util.find_spec('pip') is None:
            execution_paths.run([sys.executable, '-m', 'ensurepip', '--upgrade'], check=True)
        def install_packages(env, source):
            # The managed installer already has uv, including native SOCKS
            # support; avoid pip's optional PySocks dependency during bootstrap.
            command = ([env['FREEVIDEO_UV'], 'pip', 'install', '--python', sys.executable]
                       if env.get('FREEVIDEO_UV') else [sys.executable, '-m', 'pip', 'install'])
            result = execution_paths.run([*command, '-e', str(diffusers), '--no-deps'],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(result.stdout, flush=True)
            if result.returncode:
                raise RuntimeError(result.stdout[-2500:])
        network.package_command(network.installed_plan(), 'pypi', install_packages)
        # src is imported from this pinned checkout by paths.add_vdn(). Its
        # training-package metadata unconditionally requires FA4, whereas the
        # Engine also runs without that optional attention backend.
    print(json.dumps({'vdn_root': str(root), 'commit': head, 'packages_installed': packages,
                      'diffusers_tree': dependencies['diffusers']['patched_tree'],
                      'models_downloaded': False, 'environment_variable': 'FREEVIDEO_VDN_ROOT'}))


def prepare_diffusers(root, dependencies):
    """Build the official two-patch tree atomically; never reset a user checkout."""
    spec = dependencies['diffusers']
    target = root / dependencies['vdn']['diffusers_subdirectory']
    if not target.exists():
        checkout = root / ('engine-diffusers-' + str(time.time_ns()))
        network.clone(checkout, spec['url'], spec['base'])
        patches = sorted((root / 'diffusers_patches').glob('*.patch'))
        if len(patches) != 2:
            raise RuntimeError('Expected the two pinned VDN Diffusers patches')
        for path in patches:
            execution_paths.run(['git', 'apply', '--index', str(path)], cwd=checkout, check=True)
        tree = execution_paths.check_output(['git', 'write-tree'], cwd=checkout, text=True).strip()
        if tree != spec['patched_tree']:
            raise RuntimeError('Patched Diffusers source tree does not match the tested dependency')
        execution_paths.run(['git', '-c', 'user.name=VDN Engine dependency setup', '-c', 'user.email=engine@localhost',
                        'commit', '--quiet', '-m', 'Apply pinned OpenVDN MiniMax H3 patches'], cwd=checkout, check=True)
        checkout.rename(target)
    tree = execution_paths.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=target, text=True).strip()
    dirty = execution_paths.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=target, text=True)
    if tree != spec['patched_tree'] or dirty:
        raise RuntimeError('Existing Diffusers differs from the clean tested tree; use another VDN dependency directory.')
    return target
