import re
import ftfy


def clean_text(text: str) -> str:
    # fix non UTF-8 characters to their matching one
    cleaned = ftfy.fix_text(text)

    # fix spaces
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned
