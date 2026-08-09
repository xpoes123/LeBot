"""Safety checks for the packet upload validation. Run: python test_upload.py"""
import os
import tempfile

os.environ["PACKET_UPLOAD_DIR"] = tempfile.mkdtemp()  # isolate from /opt
import upload

PDF = b"%PDF-1.7\n...body..."
DOCX = b"PK\x03\x04" + b"\x00" * 40


def check(fn, *a, **k):
    try:
        fn(*a, **k); return None
    except upload.UploadError as e:
        return str(e)


# filename sanitization strips traversal and unsafe chars, keeps extension
assert upload.safe_name("../../etc/passwd.pdf") == "passwd.pdf"
assert upload.safe_name("My Packet #1!.PDF") == "My Packet _1_.pdf"
assert check(upload.safe_name, "evil.exe")            # bad extension rejected
assert check(upload.safe_name, "note.txt")

# magic-byte gate: a .pdf that isn't really a PDF is rejected
assert check(upload.validate_magic, "x.pdf", b"<html>") is not None
assert upload.validate_magic("x.pdf", PDF) is None
assert upload.validate_magic("x.docx", DOCX) is None

# save: happy path lands a file; required tournament + real bytes enforced
dest = upload.save_upload("AVES DE1.pdf", PDF, "AVES 2025")
assert dest.exists() and dest.read_bytes() == PDF
assert check(upload.save_upload, "x.pdf", PDF, "") == "Tournament name is required."
assert check(upload.save_upload, "x.pdf", b"", "T") == "Empty file."
assert check(upload.save_upload, "x.pdf", b"\x00" * 10, "T")  # bad magic even with .pdf ext
assert check(upload.save_upload, "x.pdf", b"%PDF" + b"0" * (upload.MAX_BYTES), "T")  # too large

# collision → unique name, never overwrite
d2 = upload.save_upload("AVES DE1.pdf", PDF, "AVES 2025")
assert d2 != dest and d2.exists()

print("ok")
