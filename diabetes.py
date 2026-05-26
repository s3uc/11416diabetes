import streamlit as st
import pandas as pd
import joblib

# [주의] 이 코드가 실행되려면 기존에 학습된 scaler와 knn_model이 미리 로드되어 있어야 합니다.
# 예시: scaler = joblib.load('scaler.pkl') / knn_model = joblib.load('knn_model.pkl')

# 웹 페이지 제목 설정
st.title("🩺 당뇨병 예측 시스템 (KNN 모델)")
st.write("아래 정보를 입력하시면 당뇨병 발병 확률을 예측합니다.")

st.divider() # 구분선

# 2열(Columns) 레이아웃으로 입력 창 배치
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("임신 횟수 입력", min_value=0, value=0, step=1)
    glucose = st.number_input("혈당 입력", min_value=0.0, value=100.0, step=1.0)
    bp = st.number_input("혈압 입력", min_value=0.0, value=70.0, step=1.0)
    skin = st.number_input("피부 두께 입력", min_value=0.0, value=20.0, step=1.0)

with col2:
    insulin = st.number_input("인슐린 입력", min_value=0.0, value=80.0, step=1.0)
    bmi = st.number_input("체질량지수(BMI) 입력", min_value=0.0, value=25.0, step=0.1)
    gene = st.number_input("당뇨 유전 지수 입력", min_value=0.0, value=0.5, step=0.01, format="%.3f")
    age = st.number_input("나이 입력", min_value=0, value=30, step=1)

st.divider()

# 예측 버튼 생성
if st.button("당뇨병 예측하기", type="primary"):
    
    # DataFrame으로 변환 (기존 컬럼명 유지)
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, gene, age]],
        columns=['임신 횟수', '혈당', '혈압', '피부 두께', '인슐린 수치', '체질량지수', '당뇨 유전 지수', '나이']
    )
    
    # 모델 예측 과정
    input_scaled = scaler.transform(input_data)
    predicted = knn_model.predict(input_scaled)
    prob = knn_model.predict_proba(input_scaled)
    
    # 결과 출력
    st.subheader("📊 예측 결과")
    
    # 당뇨 확률 계산
    diabetes_prob = prob[0][1] * 100
    
    # 메트릭(Metric) 형태로 깔끔하게 시각화
    if predicted[0] == 1:
        st.error(f"⚠️ 예측 결과: **당뇨 위험군** (결과값: {predicted[0]})")
    else:
        st.success(f"✅ 예측 결과: **정상** (결과값: {predicted[0]})")
        
    st.metric(label="당뇨병 발병 확률", value=f"{diabetes_prob:.1f} %")