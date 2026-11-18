"""Fallback runner for environments without pytest:  python -m tests.run_tests
(`pytest` is preferred and is listed in requirements.txt.)
Optional: pass name fragments to run only some files, e.g.  python -m tests.run_tests panel technical"""
import importlib, inspect, pathlib, sys, tempfile, traceback


class MP:
    def __init__(self): self._undo = []
    def setattr(self, obj, name, val):
        self._undo.append((obj, name, getattr(obj, name))); setattr(obj, name, val)
    def setenv(self, k, v):
        import os; self._undo.append((os.environ, k, os.environ.get(k, "__none__"))); os.environ[k] = v
    def undo(self):
        import os
        for o, n, v in reversed(self._undo):
            if o is os.environ:
                os.environ.pop(n, None) if v == "__none__" else os.environ.__setitem__(n, v)
            else:
                setattr(o, n, v)


def main():
    root = pathlib.Path(__file__).parent
    passed = failed = 0
    pats = sys.argv[1:]
    for f in sorted(root.glob("test_*.py")):
        if pats and not any(x in f.stem for x in pats): continue
        mod = importlib.import_module(f"tests.{f.stem}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"): continue
            kw, mp, tmp = {}, MP(), tempfile.TemporaryDirectory()
            params = inspect.signature(fn).parameters
            if "tmp_path" in params: kw["tmp_path"] = pathlib.Path(tmp.name)
            if "monkeypatch" in params: kw["monkeypatch"] = mp
            try:
                fn(**kw); passed += 1; print("PASS", f.stem, name)
            except Exception:
                failed += 1; print("FAIL", f.stem, name); traceback.print_exc()
            finally:
                mp.undo(); tmp.cleanup()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
