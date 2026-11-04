"""Every Kaggle notebook is valid Python, defines every name it uses, and runs real
modules with real arguments (a Stowaway session once failed with NameError)."""
import ast
import builtins
import importlib

from scripts.kaggle_run import FAMILIES, JOBS, notebook_source
from experiments.finetune import bases


def test_every_notebook_defines_what_it_uses():
    for job in JOBS:
        tree = ast.parse(notebook_source(job, "abc1234"))
        bound = {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        bound |= {t.id for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.For))
                  for t in ast.walk(n.targets[0] if isinstance(n, ast.Assign) else n.target)
                  if isinstance(t, ast.Name)}
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert used <= bound | set(dir(builtins)), (job, used - bound)


def test_commands_name_real_modules_and_bases():
    families = bases()
    assert set(FAMILIES) == set(families)
    for job, spec in JOBS.items():
        for cmd in spec["commands"]:
            assert cmd[0] == "-m"
            importlib.import_module(cmd[1])
            if cmd[1] == "experiments.finetune":
                assert cmd[2] in families and cmd[3] in families[cmd[2]]
        for s in spec["sources"]:
            assert s in JOBS
