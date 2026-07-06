"""Download every Science Bowl packet PDF from the public S3 bucket and extract text.

The PDFs have a real text layer, so pdftotext (no OCR) extracts them cleanly.
Idempotent: skips PDFs/txt already present. Run: python packets.py
"""
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://scibowl.s3.us-east-2.amazonaws.com"
PREFIX = "cleaned_packets/"
ROOT = os.path.join(os.path.dirname(__file__), "packets")
PDF_DIR = os.path.join(ROOT, "pdf")
TXT_DIR = os.path.join(ROOT, "txt")


def list_keys():
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        xml = urllib.request.urlopen(url).read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    return [k for k in keys if k.lower().endswith(".pdf")]


def _local(key):
    rel = key[len(PREFIX):]
    return os.path.join(PDF_DIR, rel), os.path.join(TXT_DIR, rel + ".txt")


def fetch_and_extract(key):
    pdf, txt = _local(key)
    if os.path.exists(txt):
        return "skip"
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
    os.makedirs(os.path.dirname(txt), exist_ok=True)
    if not os.path.exists(pdf):
        url = BUCKET + "/" + urllib.parse.quote(key)
        for attempt in range(4):  # S3 resets connections under load; retry with backoff
            try:
                urllib.request.urlretrieve(url, pdf)
                break
            except Exception:
                if attempt == 3:
                    return "fail:" + key
                time.sleep(1.5 * (attempt + 1))
    subprocess.run(["pdftotext", pdf, txt], check=True)
    return "done"


if __name__ == "__main__":
    keys = list_keys()
    print(f"{len(keys)} PDFs across {len(set(k.split('/')[1] for k in keys))} tournaments")
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(fetch_and_extract, keys), 1):
            if r and r.startswith("fail:"):
                fails.append(r[5:])
            if i % 50 == 0:
                print(f"  {i}/{len(keys)}")
    print(f"text in {TXT_DIR}  | failures: {len(fails)}")
    for f in fails:
        print("  FAIL", f)
