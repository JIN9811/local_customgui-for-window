# 모델 성능 비교표 설명

입력 예시:

```text
모델 성능 비교표를 설명해줘
```

이 명령은 현재 선택된 task/model 기준으로 저장 모델 성능과 후보 모델 비교표를 정리해 설명합니다. classification 모델의 경우 Accuracy, AUC, Recall, Precision, F1, MCC 같은 지표를 중심으로 보여줍니다.

## 언제 쓰나

- 여러 저장 모델 중 어떤 모델을 쓸지 비교할 때
- 현재 선택 모델의 검증 성능을 사용자에게 설명해야 할 때
- 리포트 생성 전에 표와 raw payload를 먼저 확인하고 싶을 때

## 사용 단계

1. `Model Management`에서 `classification` task를 선택합니다.
2. 사용할 모델의 `Apply`를 누릅니다.
3. 채팅 입력창에 `모델 성능 비교표를 설명해줘`를 입력합니다.
4. 답변에서 저장 모델 비교표와 자동 후보 모델 비교표를 확인합니다.
5. `원본 결과 보기 / 닫기`를 켜면 구조화된 payload를 확인할 수 있습니다.

![Selected model detail](../assets/screenshots/streamlit-prediction-controls.png)

![Performance comparison](../assets/screenshots/streamlit-diagnostics.png)

![Raw result payload](../assets/screenshots/streamlit-raw-result.png)

## 표에서 보는 지표

- `Accuracy`: 전체 예측 중 맞춘 비율
- `AUC`: class 구분 능력
- `Recall`: 실제 양성 class를 놓치지 않는 정도
- `Prec.`: 양성으로 예측한 것 중 실제 양성 비율
- `F1`: Precision과 Recall의 균형
- `MCC`: class imbalance까지 고려한 상관 지표
- `TT (Sec)`: 모델 후보 비교 시 측정된 실행 시간

## 해석할 때의 기준

- 하나의 지표만 보고 결정하지 말고 F1, AUC, MCC를 함께 봅니다.
- 저장 모델 검증 지표와 후보 모델 비교표의 상위 모델이 다를 수 있습니다.
- 검증 row 수가 작으면 운영 적용 전 별도 holdout data로 재검증하는 것이 좋습니다.
- 모델을 바꿔서 쓰려면 `Model Management`에서 해당 모델의 `Apply`를 누릅니다.

## 다음 단계

성능 비교 설명을 확인한 뒤 공유용 문서가 필요하면 [리포트 생성](report-generation.md)을 실행합니다.

## 앱 동작 기준

- 진단 요청 감지: [`streamlit_app.py`](../../streamlit_app.py)의 `is_model_diagnostic_request`
- 성능 payload 구성: [`streamlit_app.py`](../../streamlit_app.py)의 `build_model_diagnostic_payload`
- 성능 비교 렌더링: [`streamlit_app.py`](../../streamlit_app.py)의 `render_model_diagnostic_payload`
- 저장 모델 metric 로드: [`src/hd_serving/artifacts.py`](../../src/hd_serving/artifacts.py)
