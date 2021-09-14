import pytest
from pysprint import set_panic_hook

def test_panic_hook_set():
    # PyO3 injects PanicExpection to Python builtins
    # so BaseException should suffice
    with pytest.raises(BaseException):
        set_panic_hook()