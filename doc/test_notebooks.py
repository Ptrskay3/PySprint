"""
Smoke testing utility to check that the notebooks aren't broken.
"""
import ast
import atexit
from contextlib import contextmanager
import glob
import os
import pathlib
import subprocess
import sys
import traceback

HEADER = "import matplotlib\nmatplotlib.use('Agg')\n"

SKIP_NAMES = ("doc_requirements.txt",)

IN_CI = "CI" in os.environ.keys() or 'TF_BUILD' in os.environ.keys()


class MplBackendRewriter(ast.NodeTransformer):
    """
    Simply rewrite ps.interactive context managers to use non-GUI backend.
    """

    def visit_With(self, node):
        self.generic_visit(node)
        for leaf in node.items:
            if leaf.context_expr.func.attr == "interactive":
                leaf.context_expr.args = [ast.Str(s='Agg')]
        ast.fix_missing_locations(node)
        return node


class ExitHooks:
    """
    Patch for sys.excepthook because atexit shadows the exit status.
    """

    def __init__(self):
        self.exit_code = None
        self.exception = None

    def hook(self):
        self._orig_exit = sys.exit
        sys.exit = self.exit
        sys.excepthook = self.exc_handler

    def exit(self, code=0):
        self.exit_code = code
        self._orig_exit(code)

    def exc_handler(self, exc_type, exc, *args):
        self.exception = exc


@contextmanager
def redirected_output(new_stdout=None, new_stderr=None):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if new_stdout is not None:
        sys.stdout = new_stdout
    if new_stderr is not None:
        sys.stderr = new_stderr
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def exec_notebooks(test_dir, log_path='notebooktest.log'):
    # Convert notebooks to .py files
    # This shouldn't be subprocess.call, we should use nbconvert API.
    if not IN_CI:
        subprocess.call(f"jupyter nbconvert --to script {test_dir}\\hu_*.ipynb")

    # find the converted files
    test_files = glob.glob(os.path.join(test_dir, 'hu_*.py'))
    test_files.sort()

    passed = []
    failed = []

    with open(log_path, 'w') as f:

        # Capture the output and traceback and write it to the log file
        with redirected_output(new_stdout=f, new_stderr=f):

            for fname in test_files:

                print(f">>> Rewriting interactive calls in '{fname}'.")

                try:
                    with open(fname, 'r', encoding="utf-8") as fp:
                        data = fp.readlines()

                    # Use the "Agg" backend outside context managers in
                    # the converted notebook files.
                    data = HEADER + ''.join(data)
                    tree = ast.parse(data)

                    # Rewrite all the interactive context managers to
                    # ensure matplotlib is not blocking execution.
                    MplBackendRewriter().visit(tree)
                    ast.fix_missing_locations(tree)

                    print(f">>> Rewrite done.")
                    print(f">>> Executing '{fname}.'")

                    exec(
                        compile(
                            tree, filename="out", mode="exec"), {"__name__": "__main__"}
                    )

                    print(f">>> Passed {fname}.")
                    passed.append(fname)

                except Exception:

                    traceback.print_exc()
                    failed.append(fname)

    print(f">>> Passed {len(passed)}/2")
    print(f">>> Expected to fail {len(failed)}/5")
    print(f">>> Log created at {log_path}.")

    if len(passed) != 2 or len(failed) != 5:
        return 1
    print(f">>> Notebook tests passed.")
    return 0


def cleanup(test_path):
    if hooks.exit_code is not None:
        if hooks.exit_code == 0:
            pyfiles = pathlib.Path(*test_path).absolute().glob("hu_*.py")
            txtfiles = pathlib.Path(*test_path).absolute().glob("*.txt")

            print("Removing .py files:")
            for file in pyfiles:
                file.unlink()
                print(f"Removed {file}.")

            print("Removing .txt files:")
            for file in txtfiles:
                if file.name not in SKIP_NAMES:
                    file.unlink()
                    print(f"Removed {file}.")
            return 0
        else:
            print(
                f"exec_notebooks exited with non-zero ({hooks.exit_code}) exit status. Failed."
            )
    elif hooks.exception is not None:
        print(f"Process terminated by exception: {hooks.exception}")
        return 1
    else:
        return 1


hooks = ExitHooks()
hooks.hook()

if __name__ == '__main__':

    # Do a cleanup if outside CI services
    if not IN_CI:
        atexit.register(cleanup, test_path=sys.argv[1:])

    sys.exit(exec_notebooks(*sys.argv[1:]))
