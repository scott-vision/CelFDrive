"""Install the tested SAHI release without replacing Conda-managed packages."""

import subprocess
import sys


SAHI_VERSION = "0.12.6"


def main():
    """Install only the pinned SAHI wheel into the active environment."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", f"sahi=={SAHI_VERSION}"],
        check=True,
    )


if __name__ == "__main__":
    main()
