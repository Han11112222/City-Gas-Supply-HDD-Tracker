import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime

# 1. 페이지 설정
st.set_page_config(page_title="DSE Demand Forecast: HDD & CDD Analyzer", layout="wide")
st.title("🔥 DSE Demand Forecast: HDD & CDD Analyzer")

# 2. 설명 박스
st.info("""
### 💡 도일(Degree Days) 및 예측 모델 안내
- **HDD (난방도일):** $\max(18.0 - \text{평균기온}, 0)$ | 추울수록 수치 증가
- **CDD (냉방도일):** $\max(\text{평균기온} - 26.0, 0)$ | 더울수록 수치 증가
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
    
    # [오류 해결] 일자가 비어있는 행만 삭제합니다.
    df = df.dropna(subset=['일자'])
    df = df[df['일자'].dt.year >= 2010]
    
    # [마법 추가] 미래 예측을 위해 기온 결측치가 있다면, 과거 동일 날짜(월-일)의 '평년 기온'으로 자동 대체합니다!
    df['월일'] = df['일자'].dt.strftime('%m-%d')
    df['평균기온'] = df['평균기온'].fillna(df.groupby('월일')['평균기온'].transform('mean'))
    
    df = df.dropna(subset=['평균기온']) # 혹시 평년 기온도 없는 극단적 결측치만 제거
    
    df['HDD'] = df['평균기온'].apply(lambda x: max(18.0 - x, 0))
    df['CDD'] = df['평균기온'].apply(lambda x: max(x - 26.0, 0))
    df['공급량(GJ)'] = (df['공급량(MJ)'] / 1000).round(1)
    
    return df.sort_values(by='일자').drop(columns=['월일'])

# 스타일링 함수 (천단위 콤마 및 소계 하이라이트)
def style_df(df, key_col='일자'):
    def highlight_row(row):
        if row[key_col] == '[ 소계 ]':
            return ['background-color: #e6e6e6; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    # 숫자형 컬럼에 대해 천단위 콤마 및 소수점 1자리 적용 (NaN은 '-' 처리)
    format_dict = {col: lambda x: f"{x:,.1f}" if pd.notnull(x) else "-" for col in df.columns if col != key_col}
    return df.style.apply(highlight_row, axis=1).format(format_dict)

try:
    df = load_and_process_data()

    # 4. 공급량 추정 설정 섹션 (사이드바)
    st.sidebar.header("⚙️ 공급량 추정 설정")
    min_date = df['일자'].min().date()
    max_date = df['일자'].max().date()
    
    train_range = st.sidebar.date_input("학습 기간 설정", [min_date, max_date])
    
    default_pred_start = datetime.date(2026, 1, 1)
    default_pred_end = datetime.date(2026, 12, 31)
    predict_range = st.sidebar.date_input("예측 기간 설정", [default_pred_start, default_pred_end])
    
    estimate_btn = st.sidebar.button("🚀 공급량 추정 실행")

    # 1. 일별 원본 상세 데이터 (최신순)
    st.subheader("1. 일별 원본 상세 데이터 (최신순)")
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD', 'CDD']].sort_values(by='일자', ascending=False).copy()
    display_df['일자'] = display_df['일자'].dt.strftime('%Y-%m-%d')
    st.dataframe(style_df(display_df), use_container_width=True, hide_index=True)

    # 2. 일별 원본 상세 데이터 (그래프)
    st.subheader("2. 일별 원본 상세 데이터 (그래프)")
    chart_data_raw = df.set_index('일자')[['HDD', 'CDD', '공급량(GJ)']]
    st.line_chart(chart_data_raw)

    if estimate_btn:
        if len(train_range) == 2 and len(predict_range) == 2:
            # ----------------- 모델 학습 -----------------
            train_df = df[(df['일자'].dt.date >= train_range[0]) & (df['일자'].dt.date <= train_range[1])].dropna(subset=['공급량(GJ)'])
            y_train = train_df['공급량(GJ)']
            
            # [HDD 모델] 및 R2 산출
            X_train_hdd = train_df[['HDD', 'CDD']]
            hdd_model = LinearRegression().fit(X_train_hdd, y_train)
            r2_hdd = hdd_model.score(X_train_hdd, y_train)
            
            # [3차 다항식 모델] 및 R2 산출
            poly = PolynomialFeatures(degree=3)
            X_train_poly = poly.fit_transform(train_df[['평균기온']])
            poly_model = LinearRegression().fit(X_train_poly, y_train)
            r2_poly = poly_model.score(X_train_poly, y_train)
            
            # ----------------- 예측 수행 -----------------
            full_pred_df = df[(df['일자'].dt.date >= predict_range[0]) & (df['일자'].dt.date <= predict_range[1])].copy()
            X_pred_hdd = full_pred_df[['HDD', 'CDD']]
            X_pred_poly = poly.transform(full_pred_df[['평균기온']])
            
            full_pred_df['HDD_예측 공급량(GJ)'] = hdd_model.predict(X_pred_hdd)
            full_pred_df['3차 다항식 예측 공급량(GJ)'] = poly_model.predict(X_pred_poly)
            
            # 3. 공급량 추정 결과 분석 (실적 vs 예측 비교용)
            st.divider()
            st.subheader("3. 공급량 추정 결과 분석 (HDD vs 3차 다항식)")
            
            analysis_df = full_pred_df.dropna(subset=['공급량(GJ)']).copy() # 실적이 있는 데이터만
            if not analysis_df.empty:
                analysis_df['HDD_예측오차'] = analysis_df['HDD_예측 공급량(GJ)'] - analysis_df['공급량(GJ)']
                analysis_df['HDD_오차율(%)'] = (analysis_df['HDD_예측오차'] / analysis_df['공급량(GJ)']) * 100
                analysis_df['다항식_예측오차'] = analysis_df['3차 다항식 예측 공급량(GJ)'] - analysis_df['공급량(GJ)']
                analysis_df['다항식_오차율(%)'] = (analysis_df['다항식_예측오차'] / analysis_df['공급량(GJ)']) * 100
                
                cols = ['일자', '공급량(GJ)', 'HDD_예측 공급량(GJ)', 'HDD_예측오차', 'HDD_오차율(%)', 
                        '3차 다항식 예측 공급량(GJ)', '다항식_예측오차', '다항식_오차율(%)']
                
                # 일별 표
                res_daily = analysis_df[cols].sort_values(by='일자').copy()
                total_row = pd.DataFrame({
                    '일자': ['[ 소계 ]'], '공급량(GJ)': [res_daily['공급량(GJ)'].sum()],
                    'HDD_예측 공급량(GJ)': [res_daily['HDD_예측 공급량(GJ)'].sum()],
                    'HDD_예측오차': [res_daily['HDD_예측오차'].sum()],
                    'HDD_오차율(%)': [(res_daily['HDD_예측오차'].sum() / res_daily['공급량(GJ)'].sum() * 100)],
                    '3차 다항식 예측 공급량(GJ)': [res_daily['3차 다항식 예측 공급량(GJ)'].sum()],
                    '다항식_예측오차': [res_daily['다항식_예측오차'].sum()],
                    '다항식_오차율(%)': [(res_daily['다항식_예측오차'].sum() / res_daily['공급량(GJ)'].sum() * 100)]
                })
                res_daily['일자'] = res_daily['일자'].dt.strftime('%Y-%m-%d')
                st.dataframe(style_df(pd.concat([res_daily, total_row], ignore_index=True)), use_container_width=True, hide_index=True)
                
                # 월별 합산 표
                analysis_df['월'] = analysis_df['일자'].dt.strftime('%Y-%m')
                res_monthly = analysis_df.groupby('월').agg({
                    '공급량(GJ)': 'sum', 'HDD_예측 공급량(GJ)': 'sum', '3차 다항식 예측 공급량(GJ)': 'sum'
                }).reset_index()
                res_monthly['HDD_예측오차'] = res_monthly['HDD_예측 공급량(GJ)'] - res_monthly['공급량(GJ)']
                res_monthly['HDD_오차율(%)'] = (res_monthly['HDD_예측오차'] / res_monthly['공급량(GJ)']) * 100
                res_monthly['다항식_예측오차'] = res_monthly['3차 다항식 예측 공급량(GJ)'] - res_monthly['공급량(GJ)']
                res_monthly['다항식_오차율(%)'] = (res_monthly['다항식_예측오차'] / res_monthly['공급량(GJ)']) * 100
                
                st.markdown("#### 📆 월별 분석 결과")
                st.dataframe(style_df(res_monthly, '월'), use_container_width=True, hide_index=True)
                
                # [수정됨] 꺾은선 그래프 타이틀에 R2 값 명시
                st.markdown(f"#### 📈 모델별 공급량 추이 비교 (실적 포함) | **$R^2$ (결정계수) - HDD: {r2_hdd:.3f} vs 3차 다항식: {r2_poly:.3f}**")
                st.line_chart(analysis_df.set_index('일자')[['공급량(GJ)', 'HDD_예측 공급량(GJ)', '3차 다항식 예측 공급량(GJ)']])
            
            # 4. 공급량 예측 (미래 예측 전용)
            st.divider()
            st.subheader("4. 공급량 예측 (Future Forecast)")
            
            forecast_df = full_pred_df[full_pred_df['공급량(GJ)'].isna()].copy() # 실적이 없는 미래만
            if not forecast_df.empty:
                f_cols = ['일자', 'HDD_예측 공급량(GJ)', '3차 다항식 예측 공급량(GJ)']
                res_f_daily = forecast_df[f_cols].sort_values(by='일자').copy()
                
                f_total_row = pd.DataFrame({
                    '일자': ['[ 소계 ]'],
                    'HDD_예측 공급량(GJ)': [res_f_daily['HDD_예측 공급량(GJ)'].sum()],
                    '3차 다항식 예측 공급량(GJ)': [res_f_daily['3차 다항식 예측 공급량(GJ)'].sum()]
                })
                res_f_daily['일자'] = res_f_daily['일자'].dt.strftime('%Y-%m-%d')
                st.dataframe(style_df(pd.concat([res_f_daily, f_total_row], ignore_index=True)), use_container_width=True, hide_index=True)
                
                # 월별 예측 표
                forecast_df['월'] = forecast_df['일자'].dt.strftime('%Y-%m')
                res_f_monthly = forecast_df.groupby('월').agg({
                    'HDD_예측 공급량(GJ)': 'sum', '3차 다항식 예측 공급량(GJ)': 'sum'
                }).reset_index()
                st.markdown("#### 📆 월별 예측 합계")
                st.dataframe(style_df(res_f_monthly, '월'), use_container_width=True, hide_index=True)
                
                # 예측 그래프
                st.markdown("#### 📈 모델별 미래 공급량 예측 추이")
                st.line_chart(forecast_df.set_index('일자')[['HDD_예측 공급량(GJ)', '3차 다항식 예측 공급량(GJ)']])
            else:
                st.warning("선택하신 예측 기간 내에 실적이 없는 미래 날짜가 없습니다. 미래 공급량을 확인하려면 시트의 실적이 비어있는 기간을 선택해 주세요.")

except Exception as e:
    st.error(f"오류 발생: {e}")
