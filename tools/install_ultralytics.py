"""Install the tested Ultralytics release without altering Conda packages."""

import subprocess
import sys


ULTRALYTICS_VERSION = "8.3.203"


def main() -> None:
    """Install Ultralytics after Conda has resolved its native dependencies."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            f"ultralytics=={ULTRALYTICS_VERSION}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
