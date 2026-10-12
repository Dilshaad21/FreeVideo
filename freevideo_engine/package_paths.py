"""Read-only package/runtime path budgets, shared by importer and delivery."""
import zipfile

from .long_paths import io_path, path_chars

WINDOWS_PATH_LIMIT = 259
ALIAS_ROOT_RESERVE = 40
# Longest relative file in the shipped runtime inventory; fallback when a
# particular installation's portable.json cannot be read.
KNOWN_RUNTIME_PATH_CHARS = 198
# Full runtime filename: alias root + one separator + package-relative name.
PACKAGE_PATH_BUDGET = WINDOWS_PATH_LIMIT - ALIAS_ROOT_RESERVE - 1  # 218
PREFIX = 'FreeVideo-Windows/'


def longest_paths(names, limit=5):
    return [dict(path=name, chars=path_chars(name)) for name in
            sorted(set(names), key=lambda name: (-path_chars(name), name))[:limit]]


def check_package_paths(names, *, budget=PACKAGE_PATH_BUDGET):
    longest = longest_paths(names)
    if longest and longest[0]['chars'] > budget:
        raise ValueError('Package path budget exceeded (%d characters):\n%s' %
                         (budget, '\n'.join('%d: %s' % (r['chars'], r['path']) for r in longest)))
    return longest


def zip_paths(path, limit=5):
    """Central directory only; no extraction, content reads or writes."""
    with zipfile.ZipFile(io_path(path)) as archive:
        return longest_paths((row.filename.removeprefix(PREFIX)
                              for row in archive.infolist() if not row.is_dir()), limit)


def runtime_budget(packages, runtime_root_length):
    """Conservative known runtime files, including the encoder library copy.

    `packages` are inspect_archive results for the entire batch. B can replace
    the real root length with its alias length. This function never blocks an
    import; dynamic compiler/cache/output names require separate validation.
    """
    if type(runtime_root_length) is not int or runtime_root_length < 0:
        raise ValueError('runtime_root_length must be a nonnegative integer')
    from .portable import ENCODER_LIBRARY
    names = set()
    for package in packages:
        names.update(row['path'] for row in package['files'])
        marker = package.get('marker')
        if marker:
            names.add(marker)
        if package.get('kind') == 'runtime':
            configuration = package.get('configuration')
            comfy = configuration.get('comfy') if isinstance(configuration, dict) else None
            if not isinstance(comfy, str) or not comfy:
                continue
            comfy = comfy.rstrip('/') + '/'
            for row in package['files']:
                inner = row['path'][len(comfy):] if row['path'].startswith(comfy) else ''
                if inner and any(inner == item or (item.endswith('/') and inner.startswith(item))
                                 for item in ENCODER_LIBRARY):
                    names.add('engine/vendor/h3-text-encoder/' + inner)
    longest = longest_paths(names)
    relative = longest[0]['chars'] if longest else 0
    length = runtime_root_length + 1 + relative if longest else runtime_root_length
    return dict(runtime_root_length=runtime_root_length, longest=longest,
                longest_relative_chars=relative, longest_path_chars=length,
                limit=WINDOWS_PATH_LIMIT, exceeds_limit=length > WINDOWS_PATH_LIMIT,
                scope='Known package files and encoder copies; dynamic runtime paths are not bounded')
