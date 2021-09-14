import os
import platform
import pytest

no_osx = pytest.mark.skipif(platform.system() == 'Darwin', reason="There is no xvfb on osx.")
no_mpl = pytest.mark.skipif(os.environ.get("RUNNER_OS", None) == "Linux", reason="There is no xvfb setup on Linux")
