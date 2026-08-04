"""Loader for cnl-work's certification driver. We reuse it; we do not reimplement it.

`cnl-work/src/verify.py` is standalone (torch + numpy + auto_LiRPA only), so it is
loaded by path rather than vendored. That is deliberate: a vendored copy would drift
from the version that was cross-checked against JacobianOP and dReal, and the whole
argument for trusting these numbers is that this exact verifier was validated there.

For the same reason `provenance()` records the resolved path and git commit, so every
result JSON says which verifier produced it. A certificate whose verifier version is
unknown is not a certificate.

The three-way verdict (violation / unknown / certified) comes through untouched. Do
not add a code path that collapses "unknown" into "certified"; unknown means the
bound stayed loose and no counterexample was found, which is verifier incompleteness
and never evidence of safety.
"""

import importlib.util
import os
import subprocess
import sys

_CANDIDATES = [
    os.environ.get("CNL_WORK"),
    os.path.join(os.path.expanduser("~"), "cnl-work"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "cnl-work"),
    "/content/cnl-work",                     # Colab clone target
    "/content/certified-neural-lyapunov",
]

_mod = None
_root = None


def _resolve():
    global _mod, _root
    if _mod is not None:
        return _mod
    tried = []
    for root in _CANDIDATES:
        if not root:
            continue
        path = os.path.join(root, "src", "verify.py")
        tried.append(path)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("cnl_verify", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["cnl_verify"] = mod
            spec.loader.exec_module(mod)
            _mod, _root = mod, root
            return mod
    raise ImportError(
        "could not locate cnl-work/src/verify.py. Set CNL_WORK to the repo root, or "
        "on Colab clone it:\n"
        "  git clone https://github.com/sehajr-singhs/certified-neural-lyapunov "
        "/content/cnl-work\n"
        "tried:\n  " + "\n  ".join(tried)
    )


def provenance():
    _resolve()
    commit = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", _root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        pass
    dirty = None
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", _root, "status", "--porcelain", "src/verify.py"],
            stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        pass
    return dict(verifier_root=_root, verifier_commit=commit,
                verify_py_dirty=dirty,
                note="certify_box reused verbatim from cnl-work, not reimplemented")


def certify_box(*a, **kw):
    return _resolve().certify_box(*a, **kw)


def audit_verified(*a, **kw):
    return _resolve().audit_verified(*a, **kw)


def bound_ladder(*a, **kw):
    return _resolve().bound_ladder(*a, **kw)
