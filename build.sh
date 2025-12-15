#!/bin/bash
set -e

source .venv/bin/activate 
python3 -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py 
tar -czvf dist/archive.tar.gz dist/main