#!/bin/bash
set -e

source .venv/bin/activate 
# PyInstaller imports torch in a child process; without this the bundled OpenMP runtime
# aborts on affinity detection inside arm64 build containers.
KMP_AFFINITY=disabled python3 -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py
tar -czvf dist/archive.tar.gz dist/main