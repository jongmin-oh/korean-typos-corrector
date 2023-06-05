import re


def remove_words_with_pattern(text, pattern):
    result = re.sub(pattern, "", text)
    result = re.sub(r"\s+", " ", result)  # 중복 공백 제거
    result = result.strip()  # 문장 앞뒤 공백 제거

    return result


def clean(text: str) -> str:
    """
    한글, 영어만 유지

    :example
     '안녕하세요 hello 123 #@!' -> '안녕하세요hello'
    """
    jamo_patterns = "([ㄱ-ㅎㅏ-ㅣ]+)"  # 한글 단일 자음&모음제거
    special_patterns = "[-=+,#/\:$. @*\"※&%ㆍ』\\‘|\(\)\[\]\<\>`'…》.!\?]"
    text = re.sub(pattern=jamo_patterns, repl="", string=text)
    text = re.sub(pattern=special_patterns, repl="", string=text)
    text = re.sub(r"[0-9]+", "", string=text)
    text = text.strip()
    return text
