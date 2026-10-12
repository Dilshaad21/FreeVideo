"""Startup wording shared by the native launchers."""
from .compatibility import RECOVERY_SUCCESSES


def notice_text(state, zh=False):
    if state.get('recovery_disabled'):
        return ('为了稳定，生成已自动放慢；可在设置里调整。' if zh else
                'Generation is slowed automatically for stability; you can change this in Settings.')
    return (f'为了稳定，生成已自动放慢；连续完成 {RECOVERY_SUCCESSES} 次后会加快一些，也可在设置里调整。' if zh else
            f'Generation is slowed automatically for stability; it speeds up after {RECOVERY_SUCCESSES} successful generations in a row, or you can change this in Settings.')
