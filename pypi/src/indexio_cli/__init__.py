"""The indexio binary, packaged for pip. `indexio` on the command line runs it."""
import os
import subprocess
import sys

__all__ = ["main", "binary_path"]


def binary_path():
    """Path of the bundled binary for this platform, or None when this wheel
    carries none (a source install on an unsupported platform)."""
    here = os.path.dirname(os.path.abspath(__file__))
    name = "indexio.exe" if os.name == "nt" else "indexio"
    path = os.path.join(here, "bin", name)
    return path if os.path.exists(path) else None


def main():
    exe = binary_path()
    if exe is None:
        sys.stderr.write(
            "indexio-cli: this wheel carries no binary for your platform.\n"
            "Install it another way: https://github.com/IOServicesLabs/indexio#install\n"
        )
        return 1
    args = [exe] + sys.argv[1:]
    if os.name == "nt":
        return subprocess.call(args)
    os.execv(exe, args)


if __name__ == "__main__":
    sys.exit(main())
