"""Best-effort Mac hardware observations, without packages or subprocesses.

These are system-wide states and cumulative page counters, not per-process
attribution. Each optional read fails independently; unavailable libraries are
cached so a missing API is not loaded again on every sampling step.
"""
import ctypes
from functools import lru_cache
import sys
import threading
import time

from . import macos_memory


THERMAL_NAMES = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}
PRESSURE_NAMES = {1: 'normal', 2: 'warn', 4: 'critical'}
VM_COUNTERS = ('swapouts', 'compressions', 'decompressions', 'swapins')
BLOCK_SECONDS_SCOPE = ('Host block-start intervals; each block ends at the next block start, '
                       'and the last ends at the existing step-completion callback. '
                       'Includes host dispatch/waits and the final step tail; no added GPU synchronization.')
# A whole snapshot takes 1-2 ms on an M5; the wait covers a busy generation
# process, and at one sample per step it never adds a visible delay.
_READ_BUDGET_SECONDS = .05
_observing = threading.Lock()


def _optional(read):
    try:
        return read()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _foundation():
    try:
        foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.objc_getClass.argtypes, objc.objc_getClass.restype = [ctypes.c_char_p], ctypes.c_void_p
        objc.sel_registerName.argtypes, objc.sel_registerName.restype = [ctypes.c_char_p], ctypes.c_void_p
        objc.objc_autoreleasePoolPush.argtypes, objc.objc_autoreleasePoolPush.restype = [], ctypes.c_void_p
        objc.objc_autoreleasePoolPop.argtypes, objc.objc_autoreleasePoolPop.restype = [ctypes.c_void_p], None
        # Separate signatures are essential: pointers/NSInteger are 64-bit,
        # while BOOL uses only the low byte. Never mutate objc_msgSend's type.
        pointer = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        boolean = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)(('objc_msgSend', objc))
        supports = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_void_p)(('objc_msgSend', objc))
        selectors = {name: objc.sel_registerName(name) for name in (
            b'processInfo', b'thermalState', b'isLowPowerModeEnabled', b'respondsToSelector:')}
        cls = objc.objc_getClass(b'NSProcessInfo')
        if not cls or not all(selectors.values()):
            return None
        return foundation, objc, cls, selectors, pointer, integer, boolean, supports
    except Exception:
        return None


def _process_info_value(name):
    api = _foundation()
    if api is None:
        return None
    _, _, cls, selectors, pointer, integer, boolean, supports = api
    pool = api[1].objc_autoreleasePoolPush()
    try:
        info = pointer(cls, selectors[b'processInfo'])
        selector = selectors[name]
        # Avoid an Objective-C unrecognized-selector exception on older macOS.
        if not info or not supports(info, selectors[b'respondsToSelector:'], selector):
            return None
        return (boolean if name == b'isLowPowerModeEnabled' else integer)(info, selector)
    finally:
        api[1].objc_autoreleasePoolPop(pool)


@lru_cache(maxsize=1)
def _power_libraries():
    try:
        cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        io = ctypes.CDLL('/System/Library/Frameworks/IOKit.framework/IOKit')
        cf.CFRelease.argtypes, cf.CFRelease.restype = [ctypes.c_void_p], None
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        io.IOPSCopyPowerSourcesInfo.argtypes, io.IOPSCopyPowerSourcesInfo.restype = [], ctypes.c_void_p
        io.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
        io.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
        return cf, io
    except Exception:
        return None


def _power_source():
    api = _power_libraries()
    if api is None:
        return None
    cf, io = api
    info = io.IOPSCopyPowerSourcesInfo()
    if not info:
        return None
    try:
        source = io.IOPSGetProvidingPowerSourceType(info)
        if not source:
            return None
        value = ctypes.create_string_buffer(32)
        if not cf.CFStringGetCString(source, value, len(value), 0x08000100):
            return None
        # IOKit reports AC Power on desktops without batteries too. A missing
        # snapshot or an unsupported type (e.g. UPS) is unknown, never assumed AC.
        return {b'AC Power': 'ac', b'Battery Power': 'battery'}.get(value.value)
    finally:
        cf.CFRelease(info)  # The providing-source string is borrowed, not owned.


def _vm_statistics():
    lib = macos_memory._library()
    storage = (ctypes.c_uint32 * 1024)()
    count = ctypes.c_uint32(len(storage))
    host = lib.mach_host_self()
    if not host:
        return None
    try:
        if lib.host_statistics64(host, 4, storage, ctypes.byref(count)):
            return None
        if count.value * 4 < macos_memory.VMStatistics.swapouts.offset + 8:
            return None
        return macos_memory.VMStatistics.from_buffer_copy(storage)
    finally:
        task = ctypes.c_uint32.in_dll(lib, 'mach_task_self_').value
        lib.mach_port_deallocate(task, host)


def _swap_used():
    status = macos_memory.memory_status()
    total, free = status.get('swap_total_bytes'), status.get('swap_free_bytes')
    if type(total) is int and type(free) is int and 0 <= free <= total:
        return total - free
    return None


def _empty_conditions():
    return dict.fromkeys(('thermal_state', 'thermal_state_name', 'low_power_mode',
        'power_source', 'memory_pressure_level', 'memory_pressure_name', 'swap_used_bytes', *VM_COUNTERS))


def _collect(result):
    # Publish inexpensive kernel fields first. Framework initialization and the
    # IOKit powerd IPC can be slow; only those unread fields remain null on timeout.
    pressure = _optional(lambda: macos_memory._sysctl(
        'kern.memorystatus_vm_pressure_level', ctypes.c_int()).value)
    pressure = pressure if type(pressure) is int and pressure in PRESSURE_NAMES else None
    result.update(memory_pressure_level=pressure, memory_pressure_name=PRESSURE_NAMES.get(pressure))
    stats = _optional(_vm_statistics)
    result.update({name: _optional(lambda name=name: getattr(stats, name)) for name in VM_COUNTERS})
    result['swap_used_bytes'] = _optional(_swap_used)
    thermal = _optional(lambda: _process_info_value(b'thermalState'))
    thermal = thermal if type(thermal) is int and thermal in THERMAL_NAMES else None
    result.update(thermal_state=thermal, thermal_state_name=THERMAL_NAMES.get(thermal))
    low_power = _optional(lambda: _process_info_value(b'isLowPowerModeEnabled'))
    result['low_power_mode'] = low_power if type(low_power) is bool else None
    result['power_source'] = _optional(_power_source)


def runtime_conditions():
    """Small fixed-schema snapshot; None off Darwin, null fields on read errors.

    Wait at most 50 ms for native reads, including cold framework loads/powerd
    IPC. A single daemon reader may finish late; its late values never mutate a
    returned snapshot or get reused at a later step. While it is still busy,
    subsequent samples are null rather than starting unbounded reader threads.
    No retries, tensor reads, GPU fences or identifying/error text are retained.
    """
    if sys.platform != 'darwin':
        return None
    try:
        return _snapshot()
    except Exception:
        return _empty_conditions()


def _snapshot():
    result = _empty_conditions()
    deadline = time.perf_counter() + _READ_BUDGET_SECONDS
    done = threading.Event()
    if not _observing.acquire(blocking=False):
        return result
    def read():
        try:
            _collect(result)
        except Exception:
            pass
        finally:
            _observing.release()
            done.set()
    try:
        threading.Thread(target=read, name='mac-conditions', daemon=True).start()
    except Exception:
        _observing.release()
        return result
    done.wait(max(0., deadline - time.perf_counter()))
    return dict(result)
