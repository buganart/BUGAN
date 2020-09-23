#!/usr/bin/env python3
import sys
from pathlib import Path

paths = list(Path(sys.argv[1]).rglob("*.*"))

for path in paths:
    a, b, c, step, index = path.name.split("_")
    new_name = f"{a}_{b}_{c}_{int(step):06}_{index}"
    print(f"{path.name} -> {new_name}")
    path.rename(path.parent / new_name)
