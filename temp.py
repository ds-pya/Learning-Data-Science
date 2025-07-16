from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from time import sleep

# URL 리스트
url_list = [
    "https://www.google.com",
    "https://www.wikipedia.org",
    "https://www.python.org"
]

# 크롬 옵션 설정 (브라우저 띄우기 싫으면 headless=True 설정)
chrome_options = Options()
chrome_options.add_argument("--headless")  # UI 없이 실행
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# 드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# 결과 저장
results = {}

for url in url_list:
    try:
        driver.get(url)
        sleep(1)  # 페이지 로딩 대기 (필요시 조정)
        title = driver.title
        results[url] = title
    except Exception as e:
        results[url] = f"Error: {e}"

# 드라이버 종료
driver.quit()

# 결과 출력
for url, title in results.items():
    print(f"{url} -> {title}")