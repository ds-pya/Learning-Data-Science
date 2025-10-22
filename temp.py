import regex as re

# URL 정규식 (HTTP/FTP/WWW)
url_regex = re.compile(
    r'^(?:https?|ftp)://\S+$|^www\.\S+$',
    re.IGNORECASE
)

# 구분자/특수문자 정규식
# 원래의 Kotlin 버전은 너무 광범위했기 때문에 실제 의도(단어 앞뒤 구분자 제거)에 맞게 다듬음
delimiter_regex = re.compile(
    r'(?:\s*[-/\\|_~·•#@()\[\]{}]+\s*|￦)',
    re.UNICODE
)

# 필터 문자열 목록 (예시)
title_filter_list = {"네이버", "구글", "다음", "테스트", "무제"}

def filter_title(title: str) -> str:
    """텍스트 전처리: URL/특수문자/불용문 제거"""
    if not title or title.strip() == "":
        return ""

    cleaned_title = title.strip()

    # URL 패턴 제거
    if url_regex.match(cleaned_title):
        return ""

    # 완전 일치 필터링
    if cleaned_title in title_filter_list:
        return ""

    # 구분자/특수문자 제거
    cleaned_title = delimiter_regex.sub("", cleaned_title).strip()

    # 다시 필터리스트 확인
    if cleaned_title in title_filter_list:
        return ""

    return cleaned_title