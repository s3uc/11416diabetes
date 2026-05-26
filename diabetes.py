import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("머신러닝 모델 성능 비교")

if st.button("모델 학습 및 평가"):

    # 1. 독립변수 / 종속변수 분리
    X = df.drop('결과', axis=1)
    y = df['결과']

    # 2. 학습 / 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 3. 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 모델 생성
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    svm_model = SVC(kernel='rbf', C=1)

    # 5. 모델 학습
    log_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train_scaled, y_train)
    svm_model.fit(X_train_scaled, y_train)

    # 6. 예측
    log_pred = log_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test)
    knn_pred = knn_model.predict(X_test_scaled)
    svm_pred = svm_model.predict(X_test_scaled)

    # 7. 정확도 계산
    log_acc = accuracy_score(y_test, log_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    knn_acc = accuracy_score(y_test, knn_pred)
    svm_acc = accuracy_score(y_test, svm_pred)

    # 8. 결과 출력
    st.subheader("모델별 정확도")

    st.write(f"로지스틱 회귀: {log_acc:.4f}")
    st.write(f"랜덤포레스트: {rf_acc:.4f}")
    st.write(f"KNN: {knn_acc:.4f}")
    st.write(f"SVM: {svm_acc:.4f}")

    # 표 형태 출력
    result_df = pd.DataFrame({
        "모델": ["로지스틱 회귀", "랜덤포레스트", "KNN", "SVM"],
        "정확도": [log_acc, rf_acc, knn_acc, svm_acc]
    })

    st.dataframe(result_df)

    # 막대그래프
    st.bar_chart(result_df.set_index("모델"))
