import subprocess


def test_main():
    assert subprocess.check_output(["confscale", "foo", "foobar"], text=True) == "foobar\n"
