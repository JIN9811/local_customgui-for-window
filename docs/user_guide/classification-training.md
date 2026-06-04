# 분류 모델 학습

입력 예시:

```text
분류 모델 학습해줘
```

이 명령은 현재 active dataset을 classification 학습 데이터로 사용해 모델을 학습하고 저장합니다. 학습이 끝나면 새 모델 artifact가 `models/classification/<model_id>/` 아래에 저장되고, 이후 예측과 성능 비교에서 사용할 수 있습니다.

## 언제 쓰나

- `Result` target이 있는 학습용 Excel로 새 분류 모델을 만들 때
- 기존 모델 대신 방금 업로드한 데이터 기준 모델을 다시 만들 때
- 저장 모델 목록과 검증 지표를 갱신하고 싶을 때

## 사용 단계

1. `학습용 Excel 데이터셋 업로드 / 관리`에서 학습용 Excel을 업로드합니다.
2. `Training Dataset Management`의 active dataset이 원하는 파일인지 확인합니다.
3. 채팅 입력창에 `분류 모델 학습해줘`를 입력합니다.
4. 학습 메시지에서 `model_id`, `best_model`, holdout 검증 지표를 확인합니다.
5. `Model Management`에서 새 모델이 목록에 표시되는지 확인합니다.
6. 필요하면 `Apply`를 눌러 이후 예측/성능 비교에 사용할 모델을 지정합니다.

![Training dataset management](../assets/screenshots/streamlit-dataset-management.png)

![Model management](../assets/screenshots/streamlit-model-management.png)

![Selected model detail](../assets/screenshots/streamlit-prediction-controls.png)

## 학습 후 생성되는 항목

- `models/classification/<model_id>/model.joblib`: 저장 모델
- `models/classification/<model_id>/schema.json`: 예측 입력 변수 schema
- `models/classification/<model_id>/metrics.json`: 검증 지표와 후보 모델 비교 결과
- `models/classification/<model_id>/model_card.md`: 모델 요약 문서
- `figures/`: confusion matrix, classification report, AUC, PR, feature importance 등 검증 그림이 있는 경우 저장

## 화면에서 확인할 것

- `Model ID`: 새로 생성된 모델 ID
- `Model`: 선택된 알고리즘 이름
- `Metrics`: 검증 지표가 정상적으로 저장됐는지
- `Schema`: 예측에 필요한 입력 변수가 몇 개인지
- `Selected model detail`: 현재 선택 모델의 metric/schema 원문

## 다음 단계

학습이 끝나면 [저장 모델 예측](saved-classification-prediction.md)으로 새 Excel을 예측하거나, [성능 비교 설명](performance-comparison.md)으로 모델 성능을 비교할 수 있습니다.

## 앱 동작 기준

- 자연어 라우팅: [`src/hd_serving/orchestrator.py`](../../src/hd_serving/orchestrator.py)의 `학습`, `train`, `모델생성` 키워드
- 학습 실행: [`src/hd_serving/tools.py`](../../src/hd_serving/tools.py)의 `train_tool`
- classification pipeline: [`src/hd_serving/training.py`](../../src/hd_serving/training.py)의 `train_classification_model`
- 업로드 후 자동 workflow: [`streamlit_app.py`](../../streamlit_app.py)의 `run_record_workflow`
