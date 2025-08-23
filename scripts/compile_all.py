import os
import sys
import py_compile

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
errors = []

for dirpath, dirnames, filenames in os.walk(root):
    # skip virtual envs and caches
    if any(part in dirpath for part in (os.sep + '.venv', os.sep + '__pycache__', os.sep + '.pytest_cache')):
        continue
    for fn in filenames:
        if fn.endswith('.py'):
            path = os.path.join(dirpath, fn)
            try:
                py_compile.compile(path, doraise=True)
            except Exception as e:
                errors.append((path, str(e)))

if errors:
    for p, e in errors:
        print(f"SYNTAX ERROR: {p}\n{e}\n", file=sys.stderr)
    sys.exit(1)

print("OK")
