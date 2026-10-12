"""Embedded, bounded process output. Workers keep writing to durable log files."""
import codecs
import json
from pathlib import Path

from .terminal_ui import ESCAPES
from .long_paths import io_path, plain_text


class Tail:
    READ_BYTES = 64 * 1024
    DISPLAY_CHARS = 192 * 1024

    def __init__(self):
        from .diagnostics import Redactor
        self.redactor = Redactor()
        self.path = None
        self.reset()

    def reset(self):
        self.revision = getattr(self, 'revision', 0) + 1
        self.offset = 0
        self.identity = None
        self.pending = ''
        self.discard_line = False
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.text = ''

    def select(self, path):
        path = Path(path) if path else None
        if path != self.path:
            self.path = path
            self.reset()

    def clear(self):
        # Clear only the view; never truncate a worker's retained log.
        self.text = ''
        self.pending = ''
        self.decoder.reset()
        self.discard_line = False
        if self.path is not None:
            try:
                info = io_path(self.path).stat()
                self.identity = (info.st_dev, info.st_ino)
                self.offset = info.st_size
            except OSError:
                pass

    def read(self, final=False):
        if self.path is None:
            return ''
        try:
            with io_path(self.path).open('rb') as stream:
                import os
                info = os.fstat(stream.fileno())
                identity = (info.st_dev, info.st_ino)
                if self.identity != identity or info.st_size < self.offset:
                    self.reset()
                    self.identity = identity
                    # Reopening a multi-day log must not load it into GUI RAM.
                    self.offset = max(0, info.st_size - self.DISPLAY_CHARS)
                    self.discard_line = self.offset > 0
                stream.seek(self.offset)
                raw = stream.read(self.READ_BYTES)
                self.offset += len(raw)
        except OSError:
            return ''  # A log can appear after startup; try again next poll.
        value = self.pending + self.decoder.decode(raw).replace('\r', '\n')
        if final and self.offset == info.st_size:
            value += self.decoder.decode(b'', final=True)
            if value and not value.endswith('\n'):
                value += '\n'
        rows = value.split('\n')
        self.pending = rows.pop()
        if self.discard_line and rows:
            rows.pop(0)
            self.discard_line = False
        output = []
        for line in rows:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                event = None
            if isinstance(event, dict) and event.get('event') == 'freevideo_ui':
                line = ' · '.join(str(event[k]) for k in ('label', 'detail') if event.get(k))
            line = ESCAPES.sub('', line)
            line = ''.join(c for c in line if c.isprintable() or c == '\t')
            output.append(self.redactor.text(plain_text(line)))
        if len(self.pending) > self.READ_BYTES:
            self.pending = ''
            self.discard_line = True
            output.append('[Long output line omitted; original retained in the log]')
        addition = '\n'.join(output) + ('\n' if output else '')
        self.text = (self.text + addition)[-self.DISPLAY_CHARS:]
        return addition


class Terminal:
    """A collapsible panel in the main window, sharing the launcher's poll loop."""
    def __init__(self, parent, app, footer):
        from tkinter import ttk
        from . import branding as b
        self.app, self.footer = app, footer
        self.tail = Tail()
        self.sources = []
        self.follow = app.tk.BooleanVar(value=True)
        self.frame = ttk.Frame(parent, padding=(0, b.space(2), 0, 0))
        head = ttk.Frame(self.frame); head.pack(fill='x', pady=(0, b.space(1)))
        app.label(head, 'Terminal', '终端', style='Meta.TLabel').pack(side='left')
        self.selector = ttk.Combobox(head, state='readonly', width=18)
        self.selector.pack(side='left', padx=b.space(2))
        self.selector.bind('<<ComboboxSelected>>', self.select)
        app.button(head, 'Hide', '收起', self.hide).pack(side='right')
        app.button(head, 'Clear', '清屏', self.clear).pack(side='right', padx=b.space(1))
        app.button(head, 'Copy', '复制', self.copy).pack(side='right')
        follow = ttk.Checkbutton(head, text=app.t('Follow', '跟随'), variable=self.follow)
        app.labels.append((follow, 'Follow', '跟随')); follow.pack(side='right', padx=b.space(2))
        body = ttk.Frame(self.frame); body.pack(fill='both', expand=True)
        self.output = app.tk.Text(body, height=8, wrap='word', relief='flat', borderwidth=0,
            font=(b.MONO, 10), background=b.SURFACE, foreground=b.TEXT, padx=10, pady=8,
            selectbackground=b.ACCENT, selectforeground=b.TEXT, state='disabled')
        scroll = ttk.Scrollbar(body, command=self.output.yview)
        scroll.pack(side='right', fill='y'); self.output.pack(fill='both', expand=True)
        self.output.configure(yscrollcommand=scroll.set)
        self.output.bind('<Control-c>', self.copy_selection)

    def show(self):
        if not self.frame.winfo_manager():
            self.frame.pack(side='bottom', fill='x', after=self.footer)
        self.poll()

    def hide(self):
        self.frame.pack_forget()

    def toggle(self):
        self.hide() if self.frame.winfo_manager() else self.show()

    def select(self, _=None):
        index = self.selector.current()
        self.tail.select(self.sources[index][1] if 0 <= index < len(self.sources) else None)
        self.render()

    def poll(self):
        if not self.frame.winfo_manager():
            return
        sources = getattr(self.app.controller, 'terminal_sources', lambda: [])()
        labels = {'setup': self.app.t('Installation', '安装'), 'comfy': 'ComfyUI',
                  'initialize': self.app.t('Bundle setup', '整合包初始化')}
        if sources != self.sources:
            previous = self.sources
            self.sources = sources
            self.selector.configure(values=[labels.get(name, name) for name, _ in sources])
            if sources:
                # A newly started stage follows automatically; a manual choice
                # remains selected while the available sources are unchanged.
                index = next((i for i in range(len(sources)-1, -1, -1) if sources[i] not in previous), len(sources)-1)
                self.selector.current(index)
            else:
                self.selector.set('')
            self.select()
        else:
            self.selector.configure(values=[labels.get(name, name) for name, _ in sources])
        revision = self.tail.revision
        running = getattr(self.app.controller, 'terminal_running', lambda path: self.app.controller.busy)
        addition = self.tail.read(final=not running(self.tail.path))
        if revision != self.tail.revision:
            self.render()
        elif addition:
            self.append(addition)
        elif not sources:
            self.render(self.app.t('Installation and ComfyUI output will appear here.',
                '安装和 ComfyUI 的输出会显示在这里。'))

    def append(self, addition):
        self.output.configure(state='normal')
        self.output.insert('end', addition)
        excess = int(self.output.count('1.0', 'end-1c', 'chars')[0]) - self.tail.DISPLAY_CHARS
        if excess > 0:
            self.output.delete('1.0', '1.0+%dc' % excess)
        self.output.configure(state='disabled')
        if self.follow.get():
            self.output.see('end')

    def render(self, value=None):
        position = self.output.yview()[0]
        self.output.configure(state='normal')
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', self.tail.text if value is None else value)
        self.output.configure(state='disabled')
        self.output.see('end') if self.follow.get() else self.output.yview_moveto(position)

    def copy(self):
        self.app.window.clipboard_clear()
        self.app.window.clipboard_append(self.tail.text)

    def copy_selection(self, _=None):
        try:
            value = self.output.get('sel.first', 'sel.last')
        except self.app.tk.TclError:
            return 'break'
        self.app.window.clipboard_clear(); self.app.window.clipboard_append(value)
        return 'break'

    def clear(self):
        self.tail.clear()
        self.render()
