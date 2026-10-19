"""Replace bitsandbytes 0.48.2's shell-based Gaudi probe in the project venv."""
import ast
import importlib.metadata
from pathlib import Path
import sys


def patch():
    if sys.platform != "win32":
        return
    expected_venv = Path(__file__).resolve().parents[1] / ".venv-win"
    if Path(sys.prefix).resolve() != expected_venv:
        raise RuntimeError("Run this patch with the project's .venv-win Python.")
    distribution = importlib.metadata.distribution("bitsandbytes")
    if distribution.version != "0.48.2":
        raise RuntimeError("Review the compatibility patch for this bitsandbytes version.")
    path = Path(distribution.locate_file("bitsandbytes/backends/utils.py"))
    source = path.read_text(encoding="utf-8")
    marker = "# W2A: query package metadata without a localized Windows shell."
    if marker in source:
        print("bitsandbytes Windows compatibility patch already applied.")
        return
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_gaudi_sw_version")
    lines = source.splitlines(keepends=True)
    original = "".join(lines[function.lineno - 1:function.end_lineno])
    if '"pip list | grep habana-torch-plugin"' not in original:
        raise RuntimeError("Unexpected Gaudi probe; refusing to modify the dependency.")
    replacement = '''def get_gaudi_sw_version():
    # W2A: query package metadata without a localized Windows shell.
    from importlib.metadata import PackageNotFoundError, version as package_version

    try:
        return version.parse(package_version("habana-torch-plugin"))
    except PackageNotFoundError:
        return None
'''
    lines[function.lineno - 1:function.end_lineno] = [replacement]
    updated = "".join(lines)
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")
    print("Applied bitsandbytes Windows compatibility patch.")


if __name__ == "__main__":
    patch()
