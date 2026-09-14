"""Simula o cenário do deploy: app.py novo com um módulo antigo ainda na memória."""
from pathlib import Path

import universe as U


def test_stale_module_is_reloaded():
    path = Path(U.__file__)
    U._src_mtime = path.stat().st_mtime - 10  # versão "antiga" na memória
    saved = U.DEFAULT_MAX_CORR
    del U.DEFAULT_MAX_CORR  # como se o atributo ainda não existisse
    try:
        src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        code = src.split("_reload_stale_modules()\n")[0]  # só imports + função, sem rodar a interface
        ns = {"__name__": "app_reload_test"}
        exec(compile(code, "app.py", "exec"), ns)
        ns["_reload_stale_modules"]()
        assert U.DEFAULT_MAX_CORR == saved
        assert U._src_mtime == path.stat().st_mtime
    finally:
        U.DEFAULT_MAX_CORR = saved
