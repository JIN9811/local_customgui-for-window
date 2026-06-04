# Excel 요약

입력 예시:

```text
방금 올린 엑셀 요약해줘
```

이 명령은 현재 활성화된 Excel dataset을 기준으로 데이터 구조를 요약합니다. 앱은 전체 행을 LLM에 그대로 보내지 않고, 내부 요약 결과만 화면에 표시합니다.

## 언제 쓰나

- 업로드한 Excel이 앱에서 제대로 인식됐는지 확인할 때
- 컬럼명, 행/열 수, 데이터 타입, 미리보기 row를 빠르게 확인할 때
- classification/regression 학습 전에 target 컬럼 구성이 맞는지 점검할 때

## 사용 단계

1. 상단 `학습용 Excel 데이터셋 업로드 / 관리`를 펼칩니다.
2. `학습용 Excel 파일 선택`에서 `.xlsx` 또는 `.xls` 파일을 업로드합니다.
3. `Training Dataset Management`에서 `Active training dataset`이 원하는 파일인지 확인합니다.
4. 하단 채팅 입력창에 `방금 올린 엑셀 요약해줘`를 입력합니다.
5. 답변 메시지의 `원본 결과 보기 / 닫기`를 켜면 구조화된 요약 JSON을 확인할 수 있습니다.

![Excel upload](../assets/screenshots/streamlit-upload.png)

![Training dataset management](../assets/screenshots/streamlit-dataset-management.png)

## 결과에서 보는 것

- `dataset_id`: 현재 세션에서 active dataset을 식별하는 ID
- `filename`: 업로드한 Excel 파일명
- `dataset_type`: 앱이 판별한 데이터 유형
- `rows` / `columns`: 데이터 크기
- `column_list`: 컬럼 목록 preview
- `preview`: 상위 일부 row
- `target_distribution`: target 컬럼이 있는 경우 class/value 분포

## 다음 단계

요약에서 `classification_training`으로 보이면 [분류 모델 학습](classification-training.md)을 진행할 수 있습니다. 예측용 컬럼만 있는 파일이라면 [저장 모델 예측](saved-classification-prediction.md)으로 넘어가면 됩니다.

## 앱 동작 기준

- 자연어 라우팅: [`src/hd_serving/orchestrator.py`](../../src/hd_serving/orchestrator.py)의 `요약`, `summary`, `컬럼`, `미리보기` 키워드
- 요약 생성: [`src/hd_serving/tools.py`](../../src/hd_serving/tools.py)의 `get_uploaded_data_summary`
- 데이터 요약 필드: [`src/hd_serving/data_loader.py`](../../src/hd_serving/data_loader.py)의 `dataframe_summary`
