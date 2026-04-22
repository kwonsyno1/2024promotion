import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import base64
from io import BytesIO

# --- 1. 페이지 기본 설정 및 한글 폰트 설정 ---
st.set_page_config(page_title="고리 비행기 데이터 분석", page_icon="✈️", layout="wide")

# 한글 폰트 적용 (스트림릿 클라우드 및 로컬 환경 대응)
import platform
if platform.system() == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows
    plt.rc('font', family='Malgun Gothic')
else: # Linux (스트림릿 클라우드 등)
    # packages.txt에 fonts-nanum이 설치되어 있어야 작동합니다.
    plt.rc('font', family='NanumGothic') 

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


# --- 2. 헬퍼 함수 (그래프를 Base64 이미지로 변환하여 HTML 보고서에 넣기 위함) ---
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- 3. 웹앱 화면 구성 시작 ---
st.title("✈️ 고리 비행기 데이터 분석 및 머신러닝 앱")
st.markdown("선생님들을 위한 연수용 머신러닝 페이지입니다. 실험 데이터(Excel 또는 CSV)를 업로드하면 자동으로 분석 및 예측 모델을 생성합니다.")

# 1. 파일 업로드 섹션
st.header("1. 데이터 파일 업로드")
uploaded_file = st.file_uploader("데이터 파일 업로드 (지원 형식: csv, xlsx, xls)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    # 파일 확장자에 따라 다르게 읽기
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            # CSV 파일인 경우 한글 깨짐 방지를 위해 encoding 처리
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='cp949')
        
        st.success("데이터 로드 완료!")
        
        with st.expander("데이터 미리보기 (상위 5행)", expanded=True):
            st.dataframe(df.head())
            
        # 예측할 목표 변수 설정 (기본값: 거리1(cm))
        target_col = '거리1(cm)'
        
        if target_col not in df.columns:
            st.error(f"오류: 업로드한 데이터에 '{target_col}' 컬럼이 없습니다. 데이터 형식을 확인해주세요.")
            st.stop() # 실행 중단

        # --- 데이터 분석 및 시각화 섹션 ---
        st.header("2. 데이터 시각화 및 상관관계 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("변인 간 상관관계 (히트맵)")
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax_heatmap)
            st.pyplot(fig_heatmap)
            heatmap_b64 = fig_to_base64(fig_heatmap)

        with col2:
            st.subheader("주요 변인 산점도 분석")
            # 데이터에 해당 컬럼이 있는지 확인 후 그리기
            selected_cols = ['앞쪽고리지름(cm)', '전체 길이(cm)', target_col]
            available_cols = [c for c in selected_cols if c in df.columns]
            
            if len(available_cols) > 1:
                sns.set_style('whitegrid')
                plt.rc('font', family='NanumGothic') # Seaborn 스타일 적용 후 폰트 재설정
                pg = sns.pairplot(df[available_cols])
                st.pyplot(pg.fig)
                pairplot_b64 = fig_to_base64(pg.fig)
            else:
                st.warning("산점도를 그리기 위한 필수 컬럼이 부족합니다.")
                pairplot_b64 = ""

        # --- 머신러닝 모델 학습 섹션 ---
        st.header("3. 머신러닝 모델 학습 (랜덤 포레스트 회귀)")
        
        features = df.columns.drop(target_col)
        X = df[features]
        y = df[target_col]

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with st.spinner("머신러닝 모델을 학습 중입니다..."):
            # 모델 생성 및 학습
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 결과 예측 및 평가
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

        st.success(f"모델 학습 완료! (모델 설명력 R²: {r2:.4f})")
        if r2 >= 0.7:
            st.info("💡 R² 값이 0.7 이상으로 매우 신뢰할 수 있는 결과입니다.")
            
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("실제값 vs 모델 예측값")
            fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
            ax_pred.scatter(y_test, y_pred, alpha=0.5, color='blue')
            ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax_pred.set_xlabel('실제값 (Actual)')
            ax_pred.set_ylabel('모델 예측값 (Predicted)')
            st.pyplot(fig_pred)
            actual_pred_b64 = fig_to_base64(fig_pred)

        with col4:
            st.subheader("변수 중요도 (실험 영향 요소)")
            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)
            ax_imp.barh(features[indices], importances[indices], color='teal')
            ax_imp.set_xlabel('중요도')
            st.pyplot(fig_imp)
            importance_b64 = fig_to_base64(fig_imp)

        # --- 보고서 산출 섹션 ---
        st.header("4. HTML 보고서 다운로드")
        st.write("위에서 분석한 결과를 하나의 웹 문서(HTML)로 다운로드 받을 수 있습니다.")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>고리 비행기 데이터 분석 보고서</title>
            <style>
                body {{ font-family: 'Malgun Gothic', 'NanumGothic', sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 15px 0; border: 1px solid #ddd; padding: 5px; background-color: #fff; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #eee; background-color: #f8f9fa; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>고리 비행기 데이터 분석 보고서</h1>
            <div class="section">
                <h2>1. 머신러닝 학습 결과 (R²: {r2:.4f})</h2>
                <img src="data:image/png;base64,{actual_pred_b64}" alt="Actual vs Predicted">
            </div>
            <div class="section">
                <h2>2. 변수 중요도 (어떤 변인이 가장 큰 영향을 미쳤을까?)</h2>
                <img src="data:image/png;base64,{importance_b64}" alt="Importance">
            </div>
            <div class="section">
                <h2>3. 데이터 상관관계 히트맵</h2>
                <img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap">
            </div>
        </body>
        </html>
        """
        
        st.download_button(
            label="📄 분석 보고서 다운로드 (HTML)",
            data=html_content,
            file_name="고리_비행기_데이터_분석_보고서.html",
            mime="text/html"
        )

    except Exception as e:
        st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")

else:
    st.info("👆 위에 있는 버튼을 눌러 연수용 데이터(CSV 또는 Excel)를 업로드해주세요.")
