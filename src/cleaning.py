import re

HEADER_LINE = re.compile(
    r"^\s*(from|to|cc|bcc|date|subject|message-id|received|return-path|reply-to|"
    r"in-reply-to|references|mime-version|x-[\w-]+|content-[\w-]+)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

def clean_text(t: str) -> str:
    t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", t)
    t = HEADER_LINE.sub(" ", t)
    t = re.sub(r"Message-ID:\s*\S+", " ", t, flags=re.I)
    t = re.sub(r"^\s*>.*$", " ", t, flags=re.MULTILINE)
    t = re.sub(r"\[(?:link|url|verification link)\]", " URL ", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " URL ", t)
    t = re.sub(r"\S+@\S+\.\S+", " EMAIL ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t