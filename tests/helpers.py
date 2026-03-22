from contextlib import contextmanager
import importlib.util
import sys
from pathlib import Path


@contextmanager
def temporary_modules(injected_modules: dict | None = None):
    """Temporarily inject modules into ``sys.modules`` for isolated imports."""
    injected_modules = injected_modules or {}
    previous = {name: sys.modules.get(name) for name in injected_modules}
    try:
        for name, module in injected_modules.items():
            sys.modules[name] = module
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def load_module_from_path(
    module_name: str,
    path: Path,
    injected_modules: dict | None = None,
):
    """Load a module directly from a path with optional temporary imports."""
    injected_modules = injected_modules or {}
    with temporary_modules(injected_modules):
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
