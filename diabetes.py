import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # 혹은 다른 모델 사용 가능
from sklearn.metrics import accuracy_score

# --- 1. 웹 페이지 기본 설정 ---
st.set_page_config(page_title="당뇨병 예측 시스템", layout="centered")
st.title("🩺 당뇨병 위험도 예측 프로그램")
st.write("사용자의 건강 지표를 입력하여 당뇨병 유무를 예측합니다.")
st.markdown("---")

# --- 2. 데이터 로드 및 모델/스케일러 사전 학습 (캐싱 처리) ---
@st.cache_resource
def initialize_ml_components():
    """
    앱이 처음 실행될 때 데이터셋을 읽어 스케일러와 모델을 학습시킵니다.
    캐싱(@st.cache_resource)을 통해 최초 1회만 실행되므로 속도가 빠릅니다.
    """
    try:
        # 데이터셋 로드 (본인의 CSV 파일명과 일치해야 합니다)
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        # 파일이 없을 경우 앱이 크래시 나지 않도록 안내 메시지 출력 후 더미 데이터로 임시 학습
        st.error("🚨 'diabetes.csv' 파일을 찾을 수 없습니다! 깃허브 리포지토리에 파일이 포함되어 있는지 확인해주세요.")
        # 가상의 데이터 구조 생성 (에러 방지용 안전장치)
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.DataFrame(np.random.randint(20, 100, size=(100, 9)), columns=cols)
        df['Outcome'] = np.random.choice([0, 1], size=100)

    # 독립변수 / 종속변수 분리 (종속변수 컬럼명이 'Outcome' 또는 '결과'인지 확인 필요)
    target_col = 'Outcome' if 'Outcome' in df.columns else '결과'
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 스케일러 생성 및 전체 데이터로 fit (★ 이 과정이 있어야 NameError가 나지 않습니다 ★)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 모델 생성 및 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return scaler, model, X.columns.tolist()

# 모델과 스케일러 불러오기
scaler, model, feature_names = initialize_ml_components()


# --- 3. 사용자 건강 정보 입력 UI (사이드바) ---
st.sidebar.header("📝 건강 지표 입력")

# Pima 인디언 당뇨병 데이터셋 표준 기준 입력 폼
preg = st.sidebar.number_input("임신 횟수 (Pregnancies)", min_value=0, max_value=20, value=1)
glucose = st.sidebar.slider("공복 혈당 (Glucose)", 0, 200, 100)
bp = st.sidebar.slider("이완기 혈압 (BloodPressure)", 0, 130, 70)
skin = st.sidebar.slider("삼두근 피부 두께 (SkinThickness)", 0, 100, 20)
insulin = st.sidebar.slider("혈청 인슐린 (Insulin)", 0, 900, 80)
bmi = st.sidebar.slider("체질량지수 (BMI)", 0.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider("당뇨 내력 가중치 (DiabetesPedigreeFunction)", 0.0, 2.5, 0.5, 0.01)
age = st.sidebar.slider("나이 (Age)", 21, 100, 30)

# 입력 데이터를 모델에 넣을 배열 형태로 변환
input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])


# --- 4. 예측 실행 및 결과 출력 ---
st.subheader("📊 예측 결과")

if st.button("🚀 당뇨병 위험도 예측하기"):
    with st.spinner("데이터 분석 중..."):
        # [문제의 41번째 줄 해결] 앞에서 선언된 scaler 객체를 사용하여 안전하게 스케일링 진행
        input_scaled = scaler.transform(input_data)
        
        # 예측 진행
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

    # 결과 대시보드 시각화
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **당뇨병 고위험군군에 해당할 가능성이 높습니다.** (확률: {prediction_proba[0][1]*100:.1f}%)")
        st.write("정밀 검진 및 식단 관리를 권장합니다.")
    else:
        st.success(f"✅ **당뇨병 저위험군 상태입니다.** (정상 확률: {prediction_proba[0][0]*100:.1f}%)")
        st.write("지금처럼 꾸준한 건강 관리를 유지하세요!")
        
    # 세부 수치 확인
    with st.expander("🔍 입력된 스케일링 데이터 확인"):
        st.write("모델에 입력된 표준화(Scaled) 데이터 배열:")
        st.code(str(input_scaled))
