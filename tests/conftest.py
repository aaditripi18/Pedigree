"""CI sets PEDIGREE_NO_SKIPS=1: there a skipped test (or a module skipped at import) fails the run,
because a skipped test is not a passed one. Locally, skips stay skips (e.g. no C compiler)."""
import os


def pytest_sessionfinish(session, exitstatus):
    if not os.environ.get("PEDIGREE_NO_SKIPS"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        if reporter:
            reporter.write_line(f"PEDIGREE_NO_SKIPS: {len(skipped)} skipped; every test must run in CI", red=True)
        session.exitstatus = 1
