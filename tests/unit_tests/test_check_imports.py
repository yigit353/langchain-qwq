"""The import checker must use normal package imports and report failures."""

import subprocess
import sys
from pathlib import Path

CHECK_IMPORTS = Path(__file__).resolve().parents[2] / "scripts" / "check_imports.py"


def test_check_imports_supports_relative_imports(tmp_path: Path) -> None:
    package = tmp_path / "sample_package"
    package.mkdir()
    (package / "__init__.py").write_text("from .module import VALUE\n")
    (package / "module.py").write_text("from .values import VALUE\n")
    (package / "values.py").write_text("VALUE = 3\n")

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_IMPORTS),
            "sample_package/__init__.py",
            "sample_package/module.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_check_imports_reports_import_failures(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("import nonexistent_import_check_dependency\n")

    result = subprocess.run(
        [sys.executable, str(CHECK_IMPORTS), "broken.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "broken.py" in result.stdout
    assert "ModuleNotFoundError" in result.stderr
    assert "nonexistent_import_check_dependency" in result.stderr
