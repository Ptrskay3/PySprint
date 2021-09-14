# This should be enabled later
import os
import platform
import pytest

no_osx = pytest.mark.skipif(platform.system() == 'Darwin', reason="There is no xvfb on osx.")
no_mpl = pytest.mark.skipif(os.environ["RUNNER_OS"] == "Linux", reason="There is no xvfb setup on Linux")
# @pytest.fixture(scope="session", autouse=True)
# def execute_before_any_test():
#     matplotlib.use("Qt5Agg")