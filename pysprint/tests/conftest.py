# This should be enabled later
import platform
import pytest

no_osx = pytest.mark.skipif(platform.system() == 'Darwin', reason="There is no xvfb on osx.")

# @pytest.fixture(scope="session", autouse=True)
# def execute_before_any_test():
#     matplotlib.use("Qt5Agg")