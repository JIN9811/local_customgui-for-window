# Streamlit Upgrade Plan

이 문서는 `local_customgui`를 Streamlit 기반 로컬 LLM 채팅 UI로 고도화하기 위한 설계 기준이다. 기존 `app.py`의 Ollama/vLLM 호출 경로는 유지하고, Streamlit UI는 별도 진입점인 `streamlit_app.py`로 둔다.

## 참고한 공식 문서

- Streamlit 실행: https://docs.streamlit.io/develop/api-reference/cli/run
- Chat input: https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- Chat message: https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
- Session State: https://docs.streamlit.io/develop/concepts/architecture/session-state
- Status container: https://docs.streamlit.io/develop/api-reference/status/st.status
- Forms: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form

## 설계 판단

1. Streamlit 앱은 `streamlit_app.py`를 기본 진입점으로 둔다.
2. 채팅 UI는 `st.chat_input`과 `st.chat_message`를 사용한다.
3. 대화 기록과 UI 상태는 `st.session_state`에 저장한다. Streamlit은 위젯 상호작용마다 스크립트를 다시 실행하기 때문이다.
4. LLM 호출처럼 시간이 걸리는 동작은 `st.status`로 감싼다.
5. 기존 `app.py`의 `call_ollama`, `call_vllm`, `check_health`, `load_config`, `save_config`를 재사용한다.
6. 기존 dependency-free 브라우저 GUI는 유지하고, Streamlit은 별도 의존성으로만 추가한다.

## 기능 범위

- Runtime controls: backend, base URL, model, API key, temperature, max tokens, timeout, system prompt.
- Health check: Ollama `/api/tags`, vLLM `/v1/models` 확인.
- Session notes: 현재 세션에서 모델 요청에 함께 넣을 선택적 로컬 메모.
- Chat: Streamlit chat layout과 backend reasoning 표시.
- Export: transcript Markdown과 config snapshot JSON 다운로드.

## 유지할 제약

- 기존 `app.py`, `index.html`, `static/app.js`, `static/style.css` 기반 GUI는 계속 유지한다.
- Streamlit은 별도 의존성으로만 추가한다.
- Ollama는 `{base_url}/api/chat`, vLLM은 `{base_url}/chat/completions` 호출 경로를 유지한다.
- 채팅 상태는 기본적으로 Streamlit 세션 메모리에만 둔다. 파일 기반 영속 저장은 별도 요청이 있을 때 추가한다.

## 이후 개선 후보

- 모델 응답 스트리밍: 백엔드 호출을 streaming 모드로 바꾸고 `st.write_stream`으로 표시한다.
- 프롬프트 프리셋: 한국어 답변, 코드 리뷰, 문서 요약 같은 용도별 system prompt 저장.
- 세션 저장: 대화 기록을 로컬 JSONL로 저장하고 재개할 수 있게 한다.
- 모델 선택 개선: health check 결과에서 모델을 선택 목록으로 자동 반영한다.
