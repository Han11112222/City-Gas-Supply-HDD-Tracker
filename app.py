import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# 1. 페이지 설정
st.set_page_config(page_title="DSE Demand Forecast: HDD & CDD Analyzer", layout="wide")
st.title("🔥 DSE Demand Forecast: HDD & CDD Analyzer")

# 2. 설명 박스
st.info("""
### 💡 도일(Degree Days) 및 예측 모델 안내
- **HDD (난방도일):** $\\max(18.0 - \\text{평균기온}, 0)$ | 추울수록 수치 증가
- **CDD (냉방도일):** $\\max(\\text{평균기온} - 26.0, 0)$ | 더울수록 수치 증가
- **공급량 추정:** 설정한 학습 기간의 데이터를 바탕으로 HDD/CDD와 공급량 간의 상관관계를 분석하여 예측합니다.
""")

# 3. 데이터 로드 및 전처리 (속도 최적화)
@st.cache_data
def load_and_process_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    raw_df = pd.read_csv(sheet_url)
    
    df = pd.DataFrame()
    df['일자'] = pd.to_datetime(raw_df.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
    df['공급량(MJ)'] = raw_df.iloc[:, 1].astype(str).str.replace(',', '').astype(float)
    df['평균기온'] = pd.to_numeric(raw_df.iloc[:, 3], errors='coerce')
    
    # 필터링 및 계산
    df = df.dropna(subset=['일자', '평균기온', '공급량(MJ)'])
    df = df[df['일자'].dt.year >= 2010]
    
    df['HDD'] = df['평균기온'].apply(lambda x: max(18.0 - x, 0))
    df['CDD'] = df['평균기온'].apply(lambda x: max(x - 26.0, 0))
    df['공급량(GJ)'] = (df['공급량(MJ)'] / 1000).round(1)
    
    return df.sort_values(by='일자')

try:
    df = load_and_process_data()

    # 4. 공급량 추정 설정 섹션 (사이드바)
    st.sidebar.header("⚙️ 공급량 추정 설정")
    min_date = df['일자'].min().date()
    max_date = df['일자'].max().date()
    
    # 학습 기간 설정 (디폴트는 전체 기간)
    train_range = st.sidebar.date_input("학습 기간 설정", [min_date, max_date])
    
    # [수정됨] 예측 기간 디폴트 값 2026.01.01 ~ 2026.12.31 적용
    default_pred_start = datetime.date(2026, 1, 1)
    default_pred_end = datetime.date(2026, 12, 31)
    predict_range = st.sidebar.date_input("예측 기간 설정", [default_pred_start, default_pred_end])
    
    estimate_btn = st.sidebar.button("🚀 공급량 추정 실행")

    # 5. 데이터 표 및 다운로드 (최신순)
    st.subheader("📊 일별 상세 데이터 (최신순)")
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD', 'CDD']].sort_values(by='일자', ascending=False).copy()
    display_df['일자'] = display_df['일자'].dt.strftime('%Y-%m-%d')
    
    csv = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 데이터 다운로드 (CSV)", data=csv, file_name="DSE_Energy_Data.csv", mime="text/csv")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 6. 공급량 추정 결과 출력
    if estimate_btn:
        if len(train_range) == 2 and len(predict_range) == 2:
            # 학습 데이터 준비
            train_df = df[(df['일자'].dt.date >= train_range[0]) & (df['일자'].dt.date <= train_range[1])]
            X_train = train_df[['HDD', 'CDD']]
            y_train = train_df['공급량(GJ)']
            
            # 모델 학습 (선형 회귀)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 예측 데이터 준비
            pred_df = df[(df['일자'].dt.date >= predict_range[0]) & (df['일자'].dt.date <= predict_range[1])].copy()
            if not pred_df.empty:
                X_pred = pred_df[['HDD', 'CDD']]
                pred_df['예측공급량(GJ)'] = model.predict(X_pred).round(1)
                
                st.success(f"✅ 추정 완료! (학습 결정계수 $R^2$: {model.score(X_train, y_train):.3f})")
                st.write(f"**추정된 기저부하:** {model.intercept_:.1f} GJ")
                
                # 예측 결과 표
                st.subheader("📋 공급량 추정 결과")
                res_display = pred_df[['일자', '공급량(GJ)', '예측공급량(GJ)']].sort_values(by='일자', ascending=False)
                res_display['일자'] = res_display['일자'].dt.strftime('%Y-%m-%d')
                st.dataframe(res_display, use_container_width=True, hide_index=True)
            else:
                st.warning("예측 기간에 해당하는 기상 데이터가 없습니다.")

    # 7. 하단 동적 그래프
    st.subheader("📈 데이터 추이 그래프")
    chart_data = df.set_index('일자')[['HDD', 'CDD', '공급량(GJ)']]
    st.line_chart(chart_data)

except Exception as e:
    st.error(f"오류 발생: {e}")
