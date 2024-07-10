#!/bin/bash
set -e
UNAME=$(uname -s)
if [ "$UNAME" = "Linux" ]
then
    if dpkg -l python3-venv; then
    echo "python3-venv is installed, skipping setup"
    else
    echo "Installing venv on Linux"
    sudo apt-get install -y python3-venv
    fi
fi
if [ "$UNAME" = "Darwin" ]
then
echo "Installing venv on Darwin"
brew install python3-venv
fi
python3 -m venv .venv . 
source .venv/bin/activate 
pip3 install -r requirements.txt 
python3 -m PyInstaller --onefile --hidden-import="googleapiclient" src/main.py 
tar -czvf dist/archive.tar.gz dist/main