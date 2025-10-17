# util/log_nav.py
import os
import re
import json
from typing import List, Optional, Tuple, Any
import streamlit as st

LOG_DIR = "log"
FNAME_PATTERN = re.compile(r"^log_(\d{6})\.json$")  # YYMMDD

# --------- 내부 유틸 ---------
def _dir_signature(dirpath: str) -> Tuple[Tuple[str, float, int], ...]:
    """디렉터리 내 파일들의 (name, mtime, size)를 정렬한 튜플로 반환 → 캐시 키에 사용"""
    if not os.path.isdir(dirpath):
        return ()
    sig = []
    for fn in os.listdir(dirpath):
        fp = os.path.join(dirpath, fn)
        try:
            st_ = os.stat(fp)
            sig.append((fn, st_.st_mtime, st_.st_size))
        except FileNotFoundError:
            pass
    sig.sort()
    return tuple(sig)

# --------- 캐시 함수 ---------
@st.cache_data(show_spinner=False)
def list_log_dates(dir_sig: Tuple[Tuple[str, float, int], ...]) -> List[str]:
    """log 폴더에서 YYMMDD 리스트(오름차순). dir_sig를 캐시 키로 사용"""
    if not os.path.isdir(LOG_DIR):
        return []
    ys = []
    for fn in os.listdir(LOG_DIR):
        m = FNAME_PATTERN.match(fn)
        if m:
            ys.append(m.group(1))
    return sorted(set(ys))

@st.cache_data(show_spinner=False)
def load_log(yyMMdd: str, file_sig: Tuple[str, float, int]) -> Any:
    """특정 날짜의 JSON을 로드. 파일 시그니처(이름, mtime, size)를 캐시 키로 사용"""
    path = os.path.join(LOG_DIR, f"log_{yyMMdd}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _file_signature(yyMMdd: str) -> Optional[Tuple[str, float, int]]:
    path = os.path.join(LOG_DIR, f"log_{yyMMdd}.json")
    if not os.path.exists(path):
        return None
    st_ = os.stat(path)
    return (os.path.basename(path), st_.st_mtime, st_.st_size)

# --------- 표시용 포맷 ---------
def fmt_display(yyMMdd: Optional[str]) -> str:
    return "—" if not yyMMdd else yyMMdd

def fmt_human(yyMMdd: str) -> str:
    yy = int(yyMMdd[:2]); mm = int(yyMMdd[2:4]); dd = int(yyMMdd[4:6])
    return f"{2000+yy:04d}-{mm:02d}-{dd:02d}"

# --------- 공용 UI 컴포넌트 ---------
def render_date_nav(state_key: str = "log_idx") -> Optional[str]:
    """
    상단 '◀ YYMMDD ▶' + 팝오버/익스팬더 UI를 렌더하고
    현재 선택된 YYMMDD(문자열) 또는 None을 반환.
    state_key: 페이지별로 독립 상태키가 필요하면 다른 값으로 전달.
    """
    dir_sig = _dir_signature(LOG_DIR)
    dates = list_log_dates(dir_sig)

    if state_key not in st.session_state:
        st.session_state[state_key] = len(dates) - 1 if dates else -1

    def set_idx(new_idx: int):
        if 0 <= new_idx < len(dates):
            st.session_state[state_key] = new_idx

    st.markdown("### 📜 로그 탐색")
    left, center, right = st.columns([1, 3, 1])

    with left:
        disabled = (not dates) or (st.session_state[state_key] <= 0)
        if st.button("◀", use_container_width=True, disabled=disabled, key=f"{state_key}_prev"):
            set_idx(st.session_state[state_key] - 1)

    with center:
        cur_idx = st.session_state[state_key]
        current_label = fmt_display(dates[cur_idx]) if dates and cur_idx >= 0 else "—"
        if hasattr(st, "popover"):
            pop = st.popover(current_label, use_container_width=True, key=f"{state_key}_pop")
            with pop:
                st.caption("날짜 선택")
                rev_dates = list(reversed(dates))
                # 현재 선택 반영
                picked = st.radio(
                    "날짜",
                    options=rev_dates,
                    index=(0 if cur_idx < 0 else rev_dates.index(dates[cur_idx])),
                    label_visibility="collapsed",
                    key=f"{state_key}_radio",
                )
                set_idx(dates.index(picked))
        else:
            st.button(current_label, use_container_width=True, disabled=not dates, key=f"{state_key}_btn")
            with st.expander("날짜 선택", expanded=False):
                rev_dates = list(reversed(dates))
                picked = st.radio(
                    "날짜",
                    options=rev_dates,
                    index=(0 if cur_idx < 0 else rev_dates.index(dates[cur_idx])),
                    key=f"{state_key}_radio_fallback",
                )
                set_idx(dates.index(picked))

    with right:
        disabled = (not dates) or (st.session_state[state_key] >= len(dates) - 1)
        if st.button("▶", use_container_width=True, disabled=disabled, key=f"{state_key}_next"):
            set_idx(st.session_state[state_key] + 1)

    st.divider()

    if not dates or st.session_state[state_key] < 0:
        st.warning("`log/` 폴더에 `log_YYMMDD.json` 파일이 없습니다.")
        return None

    return dates[st.session_state[state_key]]

def get_log_json(yyMMdd: str) -> Optional[Any]:
    """선택된 날짜의 로그 JSON 로드(캐시 적용, 파일 변경 시 자동 무효화)"""
    sig = _file_signature(yyMMdd)
    if sig is None:
        return None
    return load_log(yyMMdd, sig)