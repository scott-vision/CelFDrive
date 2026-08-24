"""Install SAHI without allowing pip to replace Conda-managed dependencies."""

import subprocess
import sys


def main():
    """Install only the SAHI wheel into the active environment."""
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "sahi"], check=True)


if __name__ == "__main__":
    main()
