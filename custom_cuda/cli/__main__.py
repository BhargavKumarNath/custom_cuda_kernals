"""Enables `python -m custom_cuda.cli` as an alternate invocation to the
installed `custom_cuda_cli` console script.
"""

from custom_cuda.cli import main

if __name__ == "__main__":
    main()
