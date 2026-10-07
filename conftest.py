"""Make the test-suite importable from either ``pytest`` or ``python -m pytest``.

In pytest's default (prepend) import mode, the directory holding the top-most
conftest.py is added to ``sys.path``. Placing this file at the project root puts
the root on the path, so ``import triage_agent`` works for a bare ``pytest`` run
without installing the package. No fixtures are defined here.
"""
