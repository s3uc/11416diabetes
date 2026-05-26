import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("분류 모델 성능 비교")
st.write("로지스틱 회귀, 랜덤포레스트, KNN, SVM 정확도 비교")

# CSV 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    
    # 데이터 불러오기
    df = pd.read_csv(uploaded_file)

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    # 타겟 컬럼 선택
    target_col = st.selectbox(
        "종속변수(타겟) 선택",
        df.columns
    )

    if st.button("모델 학습 및 평가"):

        # 독립변수 / 종속변수 분리
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # 학습 / 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 생성
        log_model = LogisticRegression(max_iter=1000)
        rf_model = RandomForestClassifier(random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        svm_model = SVC(kernel='rbf', C=1)

        # 모델 학습
        log_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train, y_train)
        knn_model.fit(X_train_scaled, y_train)
        svm_model.fit(X_train_scaled, y_train)

        # 예측
        log_pred = log_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test)
        knn_pred = knn_model.predict(X_test_scaled)
        svm_pred = svm_model.predict(X_test_scaled)

        # 정확도 계산
        log_acc = accuracy_score(y_test, log_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        knn_acc = accuracy_score(y_test, knn_pred)
        svm_acc = accuracy_score(y_test, svm_pred)

        # 결과 출력
        st.subheader("모델 성능")

        result_df = pd.DataFrame({
            "모델": ["로지스틱 회귀", "랜덤포레스트", "KNN", "SVM"],
            "정확도": [log_acc, rf_acc, knn_acc, svm_acc]
        })

        st.dataframe(result_df)

        best_model = result_df.loc[result_df["정확도"].idxmax()]

        st.success(
            f"최고 성능 모델: {best_model['모델']} "
            f"(정확도: {best_model['정확도']:.4f})"
        )
