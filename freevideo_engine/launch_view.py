"""Children of an offline runtime start from its short root.

Where Windows limits path length, runtime_alias gives the runtime folder a short
junction. The launcher marks a child's environment with that view, and
processes.popen then spells the command, the environment and the working folder
through it, for that child and for every process it starts in turn. Nothing
else changes: settings, receipts and everything on screen keep the real paths.
"""
from collections import namedtuple
import json
import os
from pathlib import Path
import re

from .long_paths import io_path, plain

VIEW_ENV = 'FREEVIDEO_RUNTIME_VIEW'


class View(namedtuple('View', 'real root rules', defaults=(os.path,))):
    """real: the runtime folder; root: what children see; rules: os.path or ntpath."""
    __slots__ = ()

    @property
    def used(self):
        return _key(self.real, self.rules) != _key(self.root, self.rules)


def _key(path, rules):
    return rules.normcase(rules.normpath(os.fspath(path)))


def _within(path, root, rules):
    path, root = _key(path, rules), _key(root, rules)
    if not rules.isabs(path):
        return False
    try:
        return rules.commonpath((path, root)) == root
    except ValueError:  # Another drive.
        return False


def _moved(path, source, target, rules):
    relative = rules.relpath(rules.normpath(os.fspath(path)), os.fspath(source))
    return rules.normpath(rules.join(os.fspath(target), relative))


def value(item, view):
    """A path inside the runtime, spelled under the view; anything else as given."""
    if not isinstance(item, (str, os.PathLike)):
        return item
    # What resolving a deep file returns on Windows names the same file.
    path = plain(item, windows=view.rules.__name__ == 'ntpath')
    if not _within(path, view.real, view.rules):
        return item
    return _moved(path, view.real, view.root, view.rules)


def view_for(real_root) -> View:
    """May create the junction; AliasUnavailable when one is needed and cannot be made."""
    from . import runtime_alias
    real = Path(plain(real_root))
    return View(real, runtime_alias.runtime_view(real))


def for_engine(engine_root) -> View:
    """The view of the offline runtime around this engine folder, if it needs one."""
    from . import runtime_alias
    engine = Path(plain(engine_root))
    root = engine.parent
    # The first two tests read nothing from disk wherever no junction is needed.
    # The third asks by the extended path: where a junction is needed, the plain
    # one is not found beyond 259 characters.
    if (engine.name.casefold() != 'engine' or not runtime_alias.needs_view(root)
            or not io_path(root / 'portable.json').is_file()):
        return View(engine, engine)
    return view_for(root)


def for_file(path):
    """The view of the offline runtime that holds this file, or None.

    The file belongs to the nearest folder above it that has a runtime's
    manifest, whichever installation the launcher itself runs from or shows.
    """
    from . import runtime_alias
    path = Path(plain(path))
    for root in path.parents:
        try:
            if runtime_alias.needs_view(root) and io_path(root / 'portable.json').is_file():
                return view_for(root)
        except OSError:  # Includes a junction that cannot be made: opened as it is.
            return None
    return None


def entry_lines(view):
    """For the node's entry file, which runs before any module of ours can be
    imported: the source its receipt names, taken from the view the process has."""
    if not view.used:
        return ''
    return ("import os as _os\n"
            "_view = _os.environ.get(%r, '').split('|')\n"
            "if len(_view) == 2 and _source.is_relative_to(_view[1]):\n"
            "    _source = _Path(_view[0]) / _source.relative_to(_view[1])\n") % VIEW_ENV


def current(environ=None):
    parts = (os.environ if environ is None else environ).get(VIEW_ENV, '').split('|')
    if len(parts) != 2 or not all(os.path.isabs(part) for part in parts):
        return None
    return View(Path(parts[1]), Path(parts[0]))


def environment_view(env, view):
    """Mark a child's environment; processes.popen does the rest."""
    return dict(env, **{VIEW_ENV: '%s|%s' % (view.root, view.real)}) if view.used else env


def execution(command, env, cwd, view):
    """One child's command, environment and folder, spelled through the view."""
    separator = ';' if view.rules.__name__ == 'ntpath' else ':'

    def argument(item):
        if isinstance(item, str) and item.startswith('--') and '=' in item:
            flag, path = item.split('=', 1)
            return flag + '=' + value(path, view)
        return value(item, view)

    def variable(text):
        if separator not in text:
            return value(text, view)
        return separator.join(value(part, view) for part in text.split(separator))

    environment = {key: variable(text) for key, text in (os.environ if env is None else env).items()}
    environment[VIEW_ENV] = '%s|%s' % (view.root, view.real)
    return [argument(item) for item in command], environment, value(cwd, view)


def text_back(text, view):
    """What a child printed, with the real folder in place of the view."""
    if not view.used:
        return text
    windows = view.rules.__name__ == 'ntpath'
    # A JSON string doubles each backslash; the replacement keeps that form.
    pattern = r'[/\\]+'.join(re.escape(part) for part in re.split(r'[/\\]+', str(view.root)))
    real = str(view.real)

    def replace(match):
        return real.replace('\\', '\\\\') if '\\\\' in match.group(0) else real

    return re.sub(r'(?<![\w/\\:.-])' + pattern + r'(?![\w.-])', replace, text, flags=re.IGNORECASE if windows else 0)


def line_back(line, view):
    """One line of a child's output; a JSON row stays a JSON row."""
    if not view.used:
        return line
    if line.startswith('{'):
        try:
            row = json.loads(line)
        except ValueError:
            return text_back(line, view)

        def back(item):
            if isinstance(item, dict):
                return {key: back(inner) for key, inner in item.items()}
            if isinstance(item, list):
                return [back(inner) for inner in item]
            return text_back(item, view) if isinstance(item, str) else item

        changed = back(row)
        return line if changed == row else json.dumps(changed, ensure_ascii=False) + ('\n' if line.endswith('\n') else '')
    return text_back(line, view)
