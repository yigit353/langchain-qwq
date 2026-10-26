import sys
import traceback
from importlib import import_module
from pathlib import Path

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            path = Path(file).resolve()
            parts = [] if path.stem == "__init__" else [path.stem]
            root = path.parent
            while (root / "__init__.py").is_file():
                parts.insert(0, root.name)
                root = root.parent
            sys.path.insert(0, str(root))
            import_module(".".join(parts))
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
