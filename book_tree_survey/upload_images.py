#!/usr/bin/env python3
import base64
import sys
from pathlib import Path

from typeform import Client

client = Client(open("typeform_token.txt").read().strip())


def payload(path):
    return {
        "image": base64.b64encode(open(path, "rb").read()).decode("utf8"),
        "file_name": path.name,
    }


def upload(data):
    return client.request("post", "/images", data=data)


paths = sorted(list(Path(sys.argv[1]).glob("*.*")))

for path in paths:
    print(f"Uploading {path}")
    data = payload(path)
    upload(data)
