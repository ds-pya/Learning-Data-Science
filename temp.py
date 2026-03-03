import streamlit as st
import concurrent.futures
from datetime import datetime
from PIL import Image

# 향후 구현할 모델 모듈들을 임포트 (현재는 에러 방지를 위해 try-except 또는 주석 처리 가능)
try:
    from models.gemini_client import GeminiClient
    from models.gemma_local import GemmaLocalClient
except ImportError:
    st.warning("모델 모듈이 아직 생성되지 않았습니다. UI 테스트 모드로 실행됩니다.")

# ==========================================
# 1. 페이지 설정 및 세션 상태(Session State) 초기화
# ==========================================
st.set_page_config(page_title="LLM/VLM 대시보드", page_icon="🚀", layout="wide")

# 프롬프트 히스토리 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 현재 텍스트 영역에 들어갈 프롬프트 값 초기화
if "current_vlm_prompt" not in st.session_state:
    st.session_state.current_vlm_prompt = ""
if "current_llm_prompt" not in st.session_state:
    st.session_state.current_llm_prompt = ""

# ==========================================
# 2. 모델 로딩 (캐싱을 통해 한 번만 메모리에 적재)
# ==========================================
@st.cache_resource
def load_models():
    """모든 모델을 초기화하고 딕셔너리 형태로 반환합니다."""
    # 실제 구동 시에는 아래 주석을 해제하고 사용합니다.
    '''
    return {
        "gemini-3-flash": GeminiClient(),
        "gemma-3n-e2b-it": GemmaLocalClient("google/gemma-3n-e2b-it"),
        "gemma-3n-e4b-it": GemmaLocalClient("google/gemma-3n-e4b-it"),
        "gemma3-1b-it": GemmaLocalClient("google/gemma-3-1b-it", is_vlm=False)
    }
    '''
    # UI 테스트용 더미(Mock) 딕셔너리
    return {
        "gemini-3-flash": "mock",
        "gemma-3n-e2b-it": "mock",
        "gemma-3n-e4b-it": "mock",
        "gemma3-1b-it": "mock"
    }

models_dict = load_models()

# ==========================================
# 3. 사이드바: 프롬프트 히스토리
# ==========================================
with st.sidebar:
    st.header("📜 프롬프트 히스토리")
    
    if not st.session_state.history:
        st.info("아직 사용한 프롬프트가 없습니다.")
    else:
        # 최신 항목이 위에 오도록 역순 출력
        for idx, item in enumerate(reversed(st.session_state.history)):
            # 버튼 텍스트를 "시간 - 프롬프트 앞부분..." 형태로 표시
            btn_text = f"[{item['type']}] {item['time']}\n{item['prompt'][:15]}..."
            
            # 히스토리 버튼 클릭 시, 해당 탭의 입력창 값을 업데이트
            if st.button(btn_text, key=f"hist_{idx}", use_container_width=True):
                if item['type'] == "VLM":
                    st.session_state.current_vlm_prompt = item['prompt']
                else:
                    st.session_state.current_llm_prompt = item['prompt']
                st.rerun() # UI 새로고침을 통해 텍스트 영역에 값 반영

# ==========================================
# 4. 병렬 처리 함수 정의
# ==========================================
def run_inference(model_name, prompt, image_bytes=None):
    """단일 모델의 추론을 실행하는 래퍼(Wrapper) 함수"""
    try:
        model_instance = models_dict[model_name]
        
        # UI 테스트용 더미 응답
        if model_instance == "mock":
            import time; time.sleep(1.5) # 가상의 처리 시간
            return f"{model_name}의 가짜(Mock) 응답입니다.\n\n프롬프트: {prompt}"
        
        # 실제 모델 추론 호출 (VLM인지 LLM인지 구분)
        if image_bytes and hasattr(model_instance, "generate_vlm"):
            return model_instance.generate_vlm(prompt, image_bytes)
        else:
            return model_instance.generate_llm(prompt)
            
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# ==========================================
# 5. 메인 화면: 탭 구성 (VLM / LLM)
# ==========================================
st.title("🚀 멀티 모델(Gemini & Gemma) 비교 대시보드")
tab_vlm, tab_llm = st.tabs(["🖼️ VLM (비전-언어 모델)", "✍️ LLM (텍스트 전용)"])

# ----------------- VLM 탭 -----------------
with tab_vlm:
    st.subheader("이미지 및 텍스트 분석 (VLM)")
    
    col_input, col_settings = st.columns([2, 1])
    
    with col_input:
        vlm_image = st.file_uploader("이미지 첨부", type=['png', 'jpg', 'jpeg'], key="vlm_img_uploader")
        if vlm_image:
            st.image(vlm_image, caption="업로드된 이미지", use_container_width=True)
            
        # 세션 상태에 저장된 값을 default로 사용
        vlm_prompt = st.text_area("프롬프트를 입력하세요", 
                                  value=st.session_state.current_vlm_prompt, 
                                  height=150, key="vlm_text_area")

    with col_settings:
        vlm_models = ["gemini-3-flash", "gemma-3n-e2b-it", "gemma-3n-e4b-it"]
        selected_vlm = st.multiselect("모델 선택", vlm_models, default=vlm_models, key="vlm_model_sel")
        
        if st.button("VLM 모델 실행", type="primary", use_container_width=True):
            if not vlm_prompt.strip():
                st.warning("프롬프트를 입력해주세요.")
            else:
                # 히스토리 저장
                st.session_state.history.append({
                    "type": "VLM",
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prompt": vlm_prompt
                })
                
                image_bytes = vlm_image.getvalue() if vlm_image else None
                
                st.divider()
                st.write("### 📊 결과 비교")
                
                # 결과 출력을 위한 컬럼 동적 생성
                result_cols = st.columns(len(selected_vlm))
                
                # 병렬 처리 실행
                with st.spinner("모델 추론 중... (병렬 처리)"):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # 퓨처 객체 매핑
                        futures = {executor.submit(run_inference, m, vlm_prompt, image_bytes): m for m in selected_vlm}
                        
                        # 완료되는 순서대로 화면에 렌더링
                        for future in concurrent.futures.as_completed(futures):
                            model_name = futures[future]
                            col_idx = selected_vlm.index(model_name)
                            
                            with result_cols[col_idx]:
                                st.success(f"**{model_name}**")
                                st.write(future.result())

# ----------------- LLM 탭 -----------------
with tab_llm:
    st.subheader("텍스트 전용 분석 (LLM)")
    
    llm_prompt = st.text_area("프롬프트를 입력하세요", 
                              value=st.session_state.current_llm_prompt, 
                              height=150, key="llm_text_area")
    
    llm_models = ["gemini-3-flash", "gemma-3n-e2b-it", "gemma-3n-e4b-it", "gemma3-1b-it"]
    selected_llm = st.multiselect("모델 선택", llm_models, default=llm_models, key="llm_model_sel")
    
    if st.button("LLM 모델 실행", type="primary"):
        if not llm_prompt.strip():
            st.warning("프롬프트를 입력해주세요.")
        else:
            # 히스토리 저장
            st.session_state.history.append({
                "type": "LLM",
                "time": datetime.now().strftime("%H:%M:%S"),
                "prompt": llm_prompt
            })
            
            st.divider()
            st.write("### 📊 결과 비교")
            
            result_cols = st.columns(len(selected_llm))
            
            with st.spinner("텍스트 생성 중... (병렬 처리)"):
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = {executor.submit(run_inference, m, llm_prompt): m for m in selected_llm}
                    
                    for future in concurrent.futures.as_completed(futures):
                        model_name = futures[future]
                        col_idx = selected_llm.index(model_name)
                        
                        with result_cols[col_idx]:
                            st.info(f"**{model_name}**")
                            st.write(future.result())
