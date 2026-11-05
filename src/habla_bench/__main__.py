"""Allow ``python -m habla_bench``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
