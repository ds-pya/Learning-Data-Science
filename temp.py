import streamlit as st
import pandas as pd
import html as ihtml
import datetime as dt

# --- 여러분의 df 가정 ---
# df = pd.DataFrame([...])
# df["distribution_a"] = "data:image/png;base64,..."  # 혹은 base64 문자열
# df["distribution_b"] = "data:image/png;base64,..."  # 혹은 base64 문자열

TEXT_COLS  = ["source", "category", "topic"]
IMAGE_COLS = ["distribution_a", "distribution_b"]
COLUMN_ORDER = TEXT_COLS + IMAGE_COLS
COLUMN_LABELS = {
    "source": "Source",
    "category": "Category",
    "topic": "Topic",
    "distribution_a": "Distribution A",
    "distribution_b": "Distribution B",
}

def _to_img_src(val: str) -> str:
    """base64 문자열/데이터URL 모두 수용하여 최종 <img src="..."> 용 문자열 반환"""
    if val is None or val == "":
        return ""
    val = str(val)
    if val.startswith("data:image"):
        return val
    # 데이터 URL prefix가 없으면 PNG로 가정
    return "data:image/png;base64," + val

def build_table_html(df: pd.DataFrame) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    # 표 헤더
    thead = "".join(f"<th>{ihtml.escape(COLUMN_LABELS.get(c, c))}</th>" for c in COLUMN_ORDER)

    # 표 바디
    rows_html = []
    for _, row in df.iterrows():
        tds = []
        for c in COLUMN_ORDER:
            val = row.get(c, "")
            if c in IMAGE_COLS:
                src = _to_img_src(val) if pd.notna(val) else ""
                if src:
                    cell = f'<img src="{src}" style="max-height:90px; max-width:180px;">'
                else:
                    cell = ""
                tds.append(f"<td style='text-align:center; vertical-align:middle'>{cell}</td>")
            else:
                txt = "" if pd.isna(val) else ihtml.escape(str(val))
                tds.append(f"<td>{txt}</td>")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    # 전체 HTML 문서
    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Table Export</title>
<style>
  body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Apple SD Gothic Neo, Noto Sans KR, sans-serif;
    margin: 24px;
  }}
  h1 {{ margin: 0 0 4px; }}
  .meta {{ color:#666; font-size:12px; margin-bottom:16px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    table-layout: fixed;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    word-wrap: break-word;
    vertical-align: top;
  }}
  th {{
    background: #fafafa;
    text-align: left;
  }}
</style>
</head>
<body>
  <h1>Table Export</h1>
  <div class="meta">Exported: {now}</div>
  <table>
    <thead><tr>{thead}</tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
    return doc

# --- HTML 생성 및 다운로드 ---
html_doc = build_table_html(df)

st.download_button(
    "📄 표 HTML 다운로드",
    data=html_doc.encode("utf-8"),
    file_name="table_export.html",
    mime="text/html",
)

# --- PDF 생성 및 다운로드 (WeasyPrint 사용) ---
# 설치 필요: pip install weasyprint
try:
    import weasyprint
    pdf_bytes = weasyprint.HTML(string=html_doc).write_pdf()
    st.download_button(
        "🖨️ 표 PDF 다운로드",
        data=pdf_bytes,
        file_name="table_export.pdf",
        mime="application/pdf",
    )
except Exception as e:
    st.info("PDF 변환 라이브러리(weasyprint)가 설치되어 있지 않거나 환경 이슈가 있습니다. `pip install weasyprint` 후 다시 시도하세요.")
    st.exception(e)