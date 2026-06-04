# 예측 결과 리포트 생성

입력 예시:

```text
예측 결과 리포트 만들어줘
```

이 명령은 현재 선택된 모델의 성능 분석 자료를 Markdown과 PDF로 생성합니다. 최근 예측 결과가 있으면 예측용 파일 분석 섹션도 함께 반영됩니다.

## 언제 쓰나

- 모델 성능과 후보 모델 비교를 공유용 문서로 남길 때
- 예측 결과까지 포함한 검토 자료가 필요할 때
- Markdown 원문을 수정해 별도 보고서나 발표자료로 재사용하고 싶을 때

## 사용 단계

1. 먼저 [분류 모델 학습](classification-training.md) 또는 [저장 모델 예측](saved-classification-prediction.md)을 실행합니다.
2. `Model Management`에서 리포트 기준 모델이 선택되어 있는지 확인합니다.
3. 예측 결과까지 포함하려면 `저장된 분류 모델로 예측해줘`를 먼저 실행해 최신 `result_id`를 만듭니다.
4. 채팅 입력창에 `예측 결과 리포트 만들어줘`를 입력합니다.
5. 생성된 메시지 하단에서 `PDF 다운로드` 또는 `Markdown 다운로드`를 누릅니다.

![Report download](../assets/screenshots/streamlit-report-download.png)

## 리포트에 들어가는 내용

- 모델 요약
- 데이터 및 검증 조건
- 모델 성능 평가
- 저장 모델 성능 비교
- 후보 모델 비교
- 검증 시각화 해석
- 최근 예측 결과가 있는 경우 예측용 파일 분석
- 결론 및 운영 적용 시 유의사항

## 예시 산출물

- [예시 PDF 리포트](../assets/examples/classification_report_example.pdf)
- [예시 Markdown 리포트](../assets/examples/classification_report_example.md)

예시 리포트는 `classification / 20260604_224339` 모델 기준으로 생성된 산출물입니다. 실제 사용 시 파일명은 생성 시각, task, model_id에 따라 달라집니다.

## 저장 위치

앱에서 생성한 원본 리포트는 기본적으로 아래 폴더에 저장됩니다.

```text
state/reports/
```

채팅 메시지에는 PDF/Markdown 다운로드 버튼과 저장 경로가 함께 표시됩니다.

## 앱 동작 기준

- 리포트 요청 감지: [`streamlit_app.py`](../../streamlit_app.py)의 `is_report_request`
- 리포트 실행: [`streamlit_app.py`](../../streamlit_app.py)의 `run_model_report_to_chat`
- Markdown 생성/정렬: [`streamlit_app.py`](../../streamlit_app.py)의 `deterministic_report_markdown`, `ensure_report_pdf_order_content`
- PDF 저장: [`streamlit_app.py`](../../streamlit_app.py)의 `save_report_pdf`
