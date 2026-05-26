import streamlit as st
import pandas as pd

st.title("당뇨병 예측 프로그램")

# 사용자 입력
preg = st.number_input("임신 횟수", min_value=0, step=1)
glucose = st.number_input("혈당", min_value=0.0)
bp = st.number_input("혈압", min_value=0.0)
skin = st.number_input("피부 두께", min_value=0.0)
insulin = st.number_input("인슐린 수치", min_value=0.0)
bmi = st.number_input("체질량지수(BMI)", min_value=0.0)
gene = st.number_input("당뇨 유전 지수", min_value=0.0)
age = st.number_input("나이", min_value=0, step=1)

# 예측 버튼
if st.button("예측하기"):

    # DataFrame 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, gene, age]],
        columns=[
            '임신 횟수',
            '혈당',
            '혈압',
            '피부 두께',
            '인슐린 수치',
            '체질량지수',
            '당뇨 유전 지수',
            '나이'
        ]
    )

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = knn_model.predict(input_scaled)
    prob = knn_model.predict_proba(input_scaled)

    # 결과 출력
    st.subheader("예측 결과")

    if predicted[0] == 1:
        st.error("당뇨병으로 예측됩니다.")
    else:
        st.success("정상으로 예측됩니다.")

    st.write(f"당뇨 확률: **{prob[0][1]*100:.1f}%**")
