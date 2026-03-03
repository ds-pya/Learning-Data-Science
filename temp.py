import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part

class GeminiClient:
    def __init__(self, project_id=None, location="us-central1"):
        """
        GeminiClient 초기화
        :param project_id: Firebase와 연동된 Google Cloud Project ID
        :param location: Vertex AI 리전 (기본값: us-central1)
        """
        # 환경 변수에서 프로젝트 ID를 가져오거나 매개변수로 받습니다.
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            raise ValueError("GCP Project ID가 설정되지 않았습니다. 환경 변수 GOOGLE_CLOUD_PROJECT를 설정하거나 매개변수로 전달해주세요.")

        # Vertex AI 초기화
        vertexai.init(project=self.project_id, location=location)
        
        # 지정하신 Gemini 3 Flash 모델 로드
        self.model = GenerativeModel("gemini-3-flash")
        
    def generate_llm(self, prompt: str) -> str:
        """
        텍스트 전용(LLM) 추론을 수행합니다.
        app.py의 tab_llm에서 호출됩니다.
        """
        try:
            # 단순 텍스트 프롬프트 전달
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API 오류 (LLM): {str(e)}"

    def generate_vlm(self, prompt: str, image_bytes: bytes) -> str:
        """
        이미지와 텍스트(VLM) 추론을 수행합니다.
        app.py의 tab_vlm에서 호출됩니다.
        """
        try:
            # Streamlit에서 넘겨받은 이미지 바이트 데이터를 Vertex AI의 Part 객체로 변환
            # (Streamlit의 업로더는 기본적으로 bytes를 반환합니다)
            image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
            
            # 리스트 형태로 이미지와 프롬프트를 함께 전달
            response = self.model.generate_content([image_part, prompt])
            return response.text
        except Exception as e:
            return f"❌ Gemini API 오류 (VLM): {str(e)}"
