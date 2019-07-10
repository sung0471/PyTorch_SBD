#!/usr/bin/env bash
python3 main_baseline.py 2>&1 \
--auto_resume |tee results/test.log
