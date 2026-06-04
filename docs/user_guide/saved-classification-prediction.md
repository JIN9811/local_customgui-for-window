# 저장된 분류 모델로 예측

입력 예시:

```text
저장된 분류 모델로 예측해줘
```

이 명령은 현재 active Excel dataset을 선택된 classification 모델로 batch prediction합니다. 앱은 저장 모델의 `schema.json`을 기준으로 입력 컬럼을 검증한 뒤 예측 결과표를 만듭니다.

## 언제 쓰나

- 이미 저장된 classification 모델로 새 Excel 파일을 예측할 때
- 학습 직후 선택된 모델이 실제 입력 데이터에 어떻게 예측하는지 확인할 때
- 예측 결과를 result_id로 저장하고 이후 리포트에 반영하고 싶을 때

## 사용 방법 A: active dataset 바로 예측

1. 예측할 Excel을 업로드하거나, 이미 active dataset이 있는지 확인합니다.
2. `Model Management`에서 사용할 classification 모델을 `Apply`합니다.
3. 채팅 입력창에 `저장된 분류 모델로 예측해줘`를 입력합니다.
4. 답변에서 `result_id`, `model_id`, `rows`를 확인합니다.
5. 메시지를 펼치면 예측 preview와 다운로드 영역을 볼 수 있습니다.

![Model management](../assets/screenshots/streamlit-model-management.png)

## 사용 방법 B: 예측 입력 패널 사용

예측용 파일을 따로 올리거나 1-row 직접 입력을 하려면 채팅 입력창에 `분류 예측` 또는 `예측 입력`을 입력합니다. 그러면 `예측 입력` 패널이 열립니다.

1. `Prediction Task`를 `classification`으로 둡니다.
2. `Model Artifact`에서 사용할 모델을 고릅니다.
3. `예측용 Excel 업로드`로 여러 행을 예측하거나, `직접 입력`을 펼쳐 한 행의 변수값을 입력합니다.
4. 예측 실행 후 생성된 결과표와 `result_id`를 확인합니다.

![Prediction input](../assets/screenshots/streamlit-prediction-input.png)

![Direct input](../assets/screenshots/streamlit-direct-input.png)

## 결과에서 보는 것

- `prediction`: 모델이 계산한 class 값
- `prediction_label`: 사용자용 class label
- `probability_success` / `probability_failure`: 모델이 확률 출력을 지원하는 경우 표시
- `result_id`: 이후 예측 요약 또는 리포트 생성에 사용할 최근 결과 ID

## 주의할 점

- 입력 Excel의 컬럼이 저장 모델 schema와 맞아야 합니다.
- 학습용 target 컬럼이 함께 들어 있어도 예측용 입력으로 쓸 수 있지만, schema에서 제외되는 컬럼은 예측에 사용하지 않습니다.
- `저장된 분류 모델로 예측해줘`는 active dataset을 바로 예측하는 명령입니다. 새 예측용 파일만 따로 올리고 싶으면 `분류 예측`으로 입력 패널을 여는 편이 더 명확합니다.

## 앱 동작 기준

- 직접 batch 예측 라우팅: [`src/hd_serving/orchestrator.py`](../../src/hd_serving/orchestrator.py)의 `예측`, `predict`, `추론` 키워드
- 입력 패널 열기: [`streamlit_app.py`](../../streamlit_app.py)의 `prediction_input_request_task_from_text`
- 예측 실행: [`src/hd_serving/tools.py`](../../src/hd_serving/tools.py)의 `predict_tool`
- 저장 모델 추론: [`src/hd_serving/inference.py`](../../src/hd_serving/inference.py)의 `predict_batch`
