#!/bin/bash
python csetup.py
cp "build/lib.linux-x86_64-2.7/cfunctions.so" .
python Testing.py