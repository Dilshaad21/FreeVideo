"""UI-only duration ranges from this machine's complete local generations.

Keep each resolution's measured step cost separate. Only the incremental steps
scale; encoding, cold preparation, first-step overhead and decoding remain in
the measured request total. This forecast never admits a workload or selects
resource policy, and never uses prompt text or sends reports elsewhere.
"""
import json
import math
import platform
from pathlib import Path
from statistics import median
import time


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def same_device(hardware, device):
    """NVIDIA GPUs match by UUID; Macs by chip and unified memory (no UUID)."""
    if isinstance(device, dict):
        return (device.get('backend') == 'mps' and hardware.get('device_backend') == 'mps'
                and bool(device.get('name')) and hardware.get('gpu_name') == device['name']
                and hardware.get('ram_total') == device.get('unified_ram_bytes'))
    return bool(device) and hardware.get('gpu_uuid') == device


def timing_record(report, gpu_uuid, system, *, stage_details=True):
    if not isinstance(report, dict) or report.get('success') is not True:
        return None
    video = report.get('video') or {}
    if (report.get('result_cache_hit') or video.get('sampling_reused')
            or video.get('first_pass_reused') or video.get('phase') != 'complete'):
        return None
    hardware = report.get('profile', {}).get('policy', {}).get('hardware', {})
    if not same_device(hardware, gpu_uuid) or hardware.get('system') != system:
        return None
    shape = report.get('geometry', {})
    plan = report.get('sampling_plan') or shape.get('sampling_plan') or {}
    if not all(type(shape.get(k)) is int and shape[k] > 0 for k in ('width', 'height', 'frames')):
        return None
    total = report.get('request_seconds')
    if not positive(total):
        return None
    # Requests that needed recovery do not describe the cost of one normal run.
    attempts = report.get('resource_attempts') or []
    if len(attempts) > 1:
        return None
    passes = video.get('sampling_passes') if plan.get('enabled') else [video]
    if not isinstance(passes, list) or len(passes) != (2 if plan.get('enabled') else 1):
        return None
    measured = []
    for part in passes:
        times = part.get('step_seconds')
        if not isinstance(times, list) or len(times) < 2 or not all(positive(v) for v in times):
            return None
        measured.append(dict(steps=len(times), seconds=sum(times), incremental=median(times[1:])))
        if stage_details:
            measured[-1]['first'] = times[0]
    if sum(p['seconds'] for p in measured) > total * 1.01:
        return None
    base = plan.get('base_steps', video.get('config', {}).get('steps', 8))
    refine = plan.get('refine_steps', 0) if plan.get('enabled') else 0
    if measured[0]['steps'] != base or (refine and measured[1]['steps'] != refine):
        return None
    encoding = report.get('encoding') or {}
    cached = bool(encoding.get('cache_hit') or report.get('input_cache', {}).get('conditioning_hit')
                  or report.get('tuning', {}).get('conditioning_cache_hit'))
    record = dict(shape={k: shape[k] for k in ('width', 'height', 'frames')},
        two_pass=bool(plan.get('enabled')), base_steps=base, refine_steps=refine,
        passes=measured, request_seconds=total,
        conditioning_cache_hit=cached,
        text_measured=positive(encoding.get('work_seconds')) and not cached,
        sampling_seconds=video.get('sample_seconds'),
        task=video.get('config', {}).get('task', 't2va'),
        adapters=bool(report.get('profile', {}).get('engine', {}).get('loras') or
                      report.get('lora') or video.get('config', {}).get('online_lora')))
    if not stage_details:
        return record
    upscale = video.get('latent_upscale') or {}
    upscale_seconds = (sum(upscale[k] for k in ('load_seconds', 'compute_seconds'))
                       if all(type(upscale.get(k)) in (int, float) and math.isfinite(upscale[k])
                              and upscale[k] >= 0 for k in ('load_seconds', 'compute_seconds')) else None)
    stages = {'encode': encoding.get('work_seconds') if not cached else 0.,
              'load': video.get('load_seconds'),
              'pass1': measured[0]['seconds'],
              'upscale': upscale_seconds if refine else 0.,
              'pass2': measured[1]['seconds'] if refine else 0.,
              'decode_save': video.get('decode_save_seconds')}
    if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in stages.values()):
        # Time outside the reported stages (request preparation, cache checks,
        # the model handover and final checks) falls about evenly before the
        # video model starts and around saving, where the bridge counts it.
        # A cached encode has no first-stage work to attach it to.
        extra = max(0., total - sum(stages.values()))
        first = 0. if cached else extra / 2
        stages['encode'] += first
        stages['decode_save'] += extra - first
    # Optional report chronology only; legacy readers already supply newest
    # first by file mtime. Never retain a path, prompt or request identifier.
    recorded_epoch = report.get('completed_epoch', report.get('started_epoch'))
    version = (report.get('runtime_code') or {}).get('version') if isinstance(report.get('runtime_code'), dict) else None
    # Saved planner input, after encoding/idle-cache credit, is in whole bytes.
    memory = [hardware.get(k) for k in ('vram_total', 'vram_free', 'ram_total', 'ram_available')]
    in_use = None
    if all((type(v) is int and v > 0) or
           (type(v) is float and positive(v) and v.is_integer()) for v in memory):
        vram, free, ram, available = map(int, memory)
        if free <= vram and available <= ram:
            in_use = dict(vram=vram - free, ram=ram - available)
    # Only the report's own final geometry describes retained conditioning.
    # This field is evidence for capability calibration, never a live ETA input.
    counts = [shape.get(k) for k in ('reference_video_tokens', 'reference_audio_tokens')]
    reference_rows = (dict(zip(('video', 'audio'), counts))
                      if all(type(n) is int and n >= 0 for n in counts) else None)
    record.update(recorded_epoch=recorded_epoch if positive(recorded_epoch) else None,
        version=version if isinstance(version, str) and len(version) <= 64 else None,
        in_use=in_use,
        reference_rows=reference_rows,
        flow='preview' if report.get('preview') or plan.get('preview') else 'full', stages=stages,
        sampling_geometry={k: plan[k] for k in ('first', 'second') if plan.get(k)},
        references=task_has_references(video.get('config', {}).get('task', 't2va')),
        model_resident=video.get('resident_engine_cache_hit'))
    return record


def estimate(records, canvas, *, base_steps=8, refine_steps=3, two_pass=True, task='t2va', adapters=False):
    from .two_pass import validate_steps
    validate_steps(base_steps, refine_steps, two_pass)
    candidates = []
    for row in records:
        if (row['task'] != task or row['adapters'] != adapters
                or row['shape'] != {k: canvas[k] for k in ('width', 'height', 'frames')}):
            continue
        # Both passes retain their own resolution costs. Never multiply the
        # entire generation by a step ratio or confuse eight low-res steps
        # with eight full-res steps. Existing first-step warmup stays intact.
        switched = row['two_pass'] != two_pass
        if switched:
            # An observed refinement supplies full-resolution step costs. We
            # cannot infer the reverse (low-resolution cost) from one pass.
            if two_pass or not positive(row.get('sampling_seconds')):
                continue
            last = row['passes'][1]
            seconds = row['request_seconds'] - row['sampling_seconds'] + last['seconds'] + (base_steps - last['steps']) * last['incremental']
        else:
            seconds = row['request_seconds'] + (base_steps - row['base_steps']) * row['passes'][0]['incremental']
        if two_pass and not switched:
            seconds += (refine_steps - row['refine_steps']) * row['passes'][1]['incremental']
        if positive(seconds):
            candidates.append((seconds, row, switched))
    if not candidates:
        return dict(status='unavailable', reason='no_matching_local_history')
    # Prefer actual runs of the requested mode over cross-mode extrapolation.
    matched = [r for r in candidates if not r[2]]
    candidates = (matched or candidates)[:6]
    values = [v for v, _, _ in candidates]
    # Cover the observed cold/warm range as well as modest scheduling variation;
    # these are a presentation range, not a statistically calibrated interval.
    low, high = min(values) * .85, max(values) * 1.2
    if len(values) == 1:
        low, high = values[0] * .8, values[0] * 1.35
    if any(switched for _, _, switched in candidates):
        low, high = min(values) * .65, max(values) * 1.5
    encoding_known = any(r['text_measured'] for _, r, _ in candidates)
    return dict(status='estimated', seconds=median(values), low_seconds=low, high_seconds=high,
        samples=len(values), includes_text_encoding=encoding_known,
        scope='whole_request' if encoding_known else 'cached_prompt_request',
        source='local_completed_generations', capacity_guarantee=False)


_CACHE = {}


def task_has_references(task):
    return task not in (None, 't2va')


# No measured request, stage or step takes this long (about twelve days). A
# larger number in a record is damage, and would overflow the medians and sums
# of a forecast. Times extrapolated to more steps are not held to it.
_LONGEST_SECONDS = 1e6


def _believable(row):
    values = [row.get('request_seconds'), *(row.get('stages') or {}).values()]
    for part in row.get('passes') or ():
        values += [part.get(key) for key in ('seconds', 'first', 'incremental')]
    return all(value <= _LONGEST_SECONDS for value in values if type(value) in (int, float))


def _near(left, right):
    return left > 0 and abs(right / left - 1.) <= .1000001


def stage_forecast(records, request, cold_flags, prior, version=None):
    """Prefer runs of this FreeVideo version; older runs stay in use until then.

    An update can change stage times, so when any run of the current version
    matches the whole request, only runs of that version are used.
    """
    if version:
        same = [row for row in records if row.get('version') == version]
        if same:
            result = _stage_forecast(same, request, cold_flags, prior)
            if result.get('stages'):
                return result
    return _stage_forecast(records, request, cold_flags, prior)


def _stage_forecast(records, request, cold_flags, prior):
    """One whole match can seed ETA; each typical still needs two similar runs.

    Only timing_record's allowlisted successful, same-device/OS observations
    enter here. Typical step costs keep first/warm and pass identity separate.
    A partial record can teach a typical without claiming a whole local ETA.
    Use the latest three matches for each stage, not three arbitrary reports.
    Callers supply newest-first records when report chronology is unavailable.
    """
    from .stage_estimate import STAGES, canvases, token_count, step_counts
    first, second, decode = canvases(request)
    counts = step_counts(request)
    samples = {key: [] for key in (*STAGES, 'pass1_first', 'pass1_warm', 'pass2_first', 'pass2_warm')}
    whole = []
    recent = records[:48]
    if recent and all(positive(row.get('recorded_epoch')) for row in recent):
        recent = sorted(recent, key=lambda row: row['recorded_epoch'], reverse=True)
    cached = cold_flags.get('encode_cached') is True
    for row in recent:
        # A damaged record must not stop the estimate, nor leave part of itself
        # behind: its samples are kept only when the whole record computes.
        found, total = [], None
        try:
            if not _believable(row):
                continue
            same_task = row.get('task') == request.get('task', 't2va')
            same_lora = row.get('adapters', False) == bool(request.get('lora_count'))
            refs = row.get('references', task_has_references(row.get('task')))
            same_refs = refs == bool(request.get('reference_count') or task_has_references(request.get('task', 't2va')))
            r1, r2, rd = canvases(dict(row['shape'], two_pass=row['two_pass'],
                                     sampling_geometry=row.get('sampling_geometry'), flow=row.get('flow', 'full')))
            stages = row.get('stages') or {}
            if same_task and same_refs and row.get('text_measured') and positive(stages.get('encode')):
                found.append(('encode', stages['encode']))
            if same_lora and row.get('model_resident') is False and positive(stages.get('load')):
                found.append(('load', stages['load']))
            if _near(decode['width'] * decode['height'] * decode['frames'], rd['width'] * rd['height'] * rd['frames']):
                if positive(stages.get('decode_save')):
                    found.append(('decode_save', stages['decode_save']))
            similar_passes = row['two_pass'] == request.get('two_pass', True)
            for i, (target, observed, count) in enumerate(zip((first, second), (r1, r2), counts), 1):
                near = _near(token_count(target), token_count(observed))
                similar_passes = similar_passes and (near or not count)
                if not (count and same_task and same_lora and near and i <= len(row['passes'])):
                    continue
                part = row['passes'][i-1]
                seconds = part['seconds'] + (count-part['steps']) * part['incremental']
                if positive(seconds):
                    found.append((f'pass{i}', seconds))
                if positive(part.get('first')):
                    found.append((f'pass{i}_first', part['first']))
                if positive(part['incremental']):
                    found.append((f'pass{i}_warm', part['incremental']))
            if similar_passes and same_task and same_lora and positive(stages.get('upscale')):
                found.append(('upscale', stages['upscale']))
            if (request.get('flow', 'full') == row.get('flow', 'full') and similar_passes and same_task and same_lora
                    and same_refs):
                total = row['request_seconds'] + (counts[0] - row['base_steps']) * row['passes'][0]['incremental']
                if counts[1]:
                    total += (counts[1] - row['refine_steps']) * row['passes'][1]['incremental']
                # Reusing the prompt cache removes the first stage and little
                # else, so a run in the other state can still describe this
                # request: without its first stage, or with one added below.
                hit = bool(row.get('conditioning_cache_hit', False))
                if cached and not hit:
                    total = total - stages['encode'] if positive(stages.get('encode')) else None
                total = (total, hit and not cached, hit == cached) if total is not None and positive(total) else None
        except (LookupError, TypeError, ValueError, ArithmeticError, AttributeError):
            continue
        for key, value in found:
            samples[key].append(value)
        if total is not None:
            whole.append(total)
    samples = {key: values[:3] for key, values in samples.items()}
    # Runs in the request's own prompt-cache state come first; converted runs
    # are used only when there is none.
    whole = ([item for item in whole if item[2]] or whole)[:3]
    typical = {}
    for key, values in samples.items():
        if len(values) < 2:
            continue
        seconds = median(values)
        unit = 5 if seconds < 60 else 10 if seconds < 600 else 60
        rounded = math.floor(seconds / unit + .5) * unit
        if rounded >= 10:
            typical[key] = rounded
    result = dict(samples=len(whole), typical=typical)
    if whole:
        weights = {key: (median(samples[key]) if samples[key] else prior[key]) for key in STAGES}
        if cached:
            weights['encode'] = 0.
        # A run that reused the prompt cache lacks the encode this request will do.
        seconds = median([total + (weights['encode'] if encodes else 0.) for total, encodes, _ in whole])
        # Preserve whole-request observed overhead, including sparse old reports.
        # It is mostly request preparation before the video model starts (the
        # first stage in the bridge), so it is not spread over later stages.
        # A cached encode has no first-stage work, so its share goes to saving.
        extra = seconds - sum(weights.values())
        if extra >= 0:
            stage = 'decode_save' if cached else 'encode'
            stages = dict(weights, **{stage: weights[stage] + extra})
        else:
            stages = {k: v * seconds / max(sum(weights.values()), .001) for k, v in weights.items()}
        result.update(seconds=seconds, stages=stages)
        # Unrounded warm step costs, so live steps can scale this machine's history.
        result['warm'] = {key: median(samples[key + '_warm']) for key in ('pass1', 'pass2')
                          if samples[key + '_warm']}
    return result


def local_records(output_directory, gpu_uuid, *, system=None, stage_details=False, refresh=False):
    """Bounded recent JSON reads, cached between slider changes; never open MP4s."""
    system = system or platform.system()
    root = Path(output_directory).resolve() / 'FreeVideo'
    key = (str(root), json.dumps(gpu_uuid, sort_keys=True), system)
    if stage_details:
        key += (True,)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and not refresh and now - cached[0] < 15:
        return cached[1]
    paths = []
    for path in root.glob('*/*/video.request.json'):
        try:
            if path.resolve().is_relative_to(root) and path.stat().st_size <= 4 * 1024 * 1024:
                paths.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    records = []
    for _, path in sorted(paths, reverse=True)[:48]:
        try:
            with path.open('rb') as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                continue
            row = timing_record(json.loads(data), gpu_uuid, system, stage_details=stage_details)
            if row:
                records.append(row)
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
    if not stage_details:
        _CACHE.clear()
    _CACHE[key] = (now, records)
    return records
