# Model Card - regression

## 목적
HD현대일렉트릭 고압차단기 수치형 형상/시험 변수 기반 `regression` 모델 artifact입니다.

## 데이터
- model family: `형상/시험 변수 기반 고압차단기 성능 예측`
- Excel 설계모델명: `N/A`
- source file: `Reg_extracted.xlsx`
- feature count: 56
- target: `TRVmax[kV]`

## Preprocessing
- Notebook 기준 `dropna()` 수행.
- Classification은 `Result == 1.0 -> label 1`, 그 외 `0`.
- Regression은 `TRVmax[kV]`를 target으로 사용.
- Ignored columns: ['Time', 'Result', 'CZM', 'Test', 'TRVmax[kV]']

## Holdout Metrics
- index: rf
- Model: Random Forest Regressor
- MAE: 3.2351
- MSE: 17.1963
- RMSE: 4.1144
- R2: 0.5137
- RMSLE: 0.2791
- MAPE: 0.2698
- TT (Sec): 0.031

## 제한사항
- 데이터 버전과 preprocessing에 따라 성능이 달라질 수 있습니다.
- LLM은 예측값을 임의 생성하지 않고 저장된 model artifact와 deterministic tool 결과만 설명해야 합니다.
