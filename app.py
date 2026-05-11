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
    
    # [수정됨] 미래 예측을 위해 공급량 결측치(빈칸) 데이터도 삭제하지 않고 유지합니다.
    df = df.dropna(subset=['일자', '평균기온'])
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
    
    train_range = st.sidebar.date_input("학습 기간 설정", [min_date, max_date])
    
    default_pred_start = datetime.date(2026, 1, 1)
    default_pred_end = datetime.date(2026, 12, 31)
    predict_range = st.sidebar.date_input("예측 기간 설정", [default_pred_start, default_pred_end])
    
    estimate_btn = st.sidebar.button("🚀 공급량 추정 실행")

    # 5. 데이터 표 및 다운로드 (상단 - 최신순)
    st.subheader("📊 일별 원본 상세 데이터 (최신순)")
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD', 'CDD']].sort_values(by='일자', ascending=False).copy()
    display_df['일자'] = display_df['일자'].dt.strftime('%Y-%m-%d')
    
    csv_raw = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 원본 데이터 다운로드 (CSV)", data=csv_raw, file_name="DSE_Raw_Data.csv", mime="text/csv")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 6. 중간 동적 그래프
    st.subheader("📈 데이터 추이 그래프")
    chart_data = df.set_index('일자')[['HDD', 'CDD', '공급량(GJ)']]
    st.line_chart(chart_data)

    # 7. 최하단: 공급량 추정 결과 출력
    if estimate_btn:
        if len(train_range) == 2 and len(predict_range) == 2:
            # ----------------- 모델 학습 -----------------
            # 학습 시에는 실적이 있는 데이터만 사용합니다.
            train_df = df[(df['일자'].dt.date >= train_range[0]) & (df['일자'].dt.date <= train_range[1])].dropna(subset=['공급량(GJ)'])
            
            # [모델 A] HDD/CDD 다중 선형 회귀
            X_train_hdd = train_df[['HDD', 'CDD']]
            y_train = train_df['공급량(GJ)']
            hdd_model = LinearRegression()
            hdd_model.fit(X_train_hdd, y_train)
            
            # [모델 B] 기온 기반 3차 다항식 회귀
            poly = PolynomialFeatures(degree=3)
            X_train_poly = poly.fit_transform(train_df[['평균기온']])
            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, y_train)
            
            # ----------------- 예측 수행 -----------------
            # 예측 시에는 실적이 없는 미래 날짜도 포함합니다.
            pred_df = df[(df['일자'].dt.date >= predict_range[0]) & (df['일자'].dt.date <= predict_range[1])].copy()
            
            if not pred_df.empty:
                st.divider()
                st.subheader("📋 공급량 추정 결과 분석 (HDD vs 3차 다항식)")
                
                # 예측값 산출
                X_pred_hdd = pred_df[['HDD', 'CDD']]
                X_pred_poly = poly.transform(pred_df[['평균기온']])
                
                pred_df['공급량 실적(GJ)'] = pred_df['공급량(GJ)']
                pred_df['HDD_예측 공급량(GJ)'] = hdd_model.predict(X_pred_hdd)
                pred_df['3차 다항식 예측 공급량(GJ)'] = poly_model.predict(X_pred_poly)
                
                # 오차 및 오차율 계산 (실적이 없는 미래는 자연스럽게 NaN 처리됨)
                pred_df['HDD_예측오차(GJ)'] = pred_df['HDD_예측 공급량(GJ)'] - pred_df['공급량 실적(GJ)']
                pred_df['HDD_오차율(%)'] = (pred_df['HDD_예측오차(GJ)'] / pred_df['공급량 실적(GJ)']) * 100
                
                pred_df['다항식_예측오차(GJ)'] = pred_df['3차 다항식 예측 공급량(GJ)'] - pred_df['공급량 실적(GJ)']
                pred_df['다항식_오차율(%)'] = (pred_df['다항식_예측오차(GJ)'] / pred_df['공급량 실적(GJ)']) * 100
                
                daily_cols = [
                    '일자', '공급량 실적(GJ)', 
                    'HDD_예측 공급량(GJ)', 'HDD_예측오차(GJ)', 'HDD_오차율(%)',
                    '3차 다항식 예측 공급량(GJ)', '다항식_예측오차(GJ)', '다항식_오차율(%)'
                ]
                res_display = pred_df.sort_values(by='일자', ascending=True).copy()
                
                # ----------------- 소계 계산 -----------------
                # 오차율 소계는 실적이 존재하는 기간에 한해서만 정확히 비교 계산합니다.
                valid_actuals = res_display.dropna(subset=['공급량 실적(GJ)'])
                sum_actual = valid_actuals['공급량 실적(GJ)'].sum()
                sum_hdd_pred_valid = valid_actuals['HDD_예측 공급량(GJ)'].sum()
                sum_poly_pred_valid = valid_actuals['3차 다항식 예측 공급량(GJ)'].sum()
                
                total_actual = res_display['공급량 실적(GJ)'].sum(min_count=1)
                total_hdd_pred = res_display['HDD_예측 공급량(GJ)'].sum()
                total_poly_pred = res_display['3차 다항식 예측 공급량(GJ)'].sum()
                
                total_hdd_err = sum_hdd_pred_valid - sum_actual if sum_actual > 0 else np.nan
                total_hdd_err_pct = (total_hdd_err / sum_actual * 100) if sum_actual > 0 else np.nan
                total_poly_err = sum_poly_pred_valid - sum_actual if sum_actual > 0 else np.nan
                total_poly_err_pct = (total_poly_err / sum_actual * 100) if sum_actual > 0 else np.nan
                
                res_display['일자'] = res_display['일자'].dt.strftime('%Y-%m-%d')
                res_display = res_display[daily_cols]
                
                subtotal_row = pd.DataFrame({
                    '일자': ['[ 소계 ]'],
                    '공급량 실적(GJ)': [total_actual],
                    'HDD_예측 공급량(GJ)': [total_hdd_pred],
                    'HDD_예측오차(GJ)': [total_hdd_err],
                    'HDD_오차율(%)': [total_hdd_err_pct],
                    '3차 다항식 예측 공급량(GJ)': [total_poly_pred],
                    '다항식_예측오차(GJ)': [total_poly_err],
                    '다항식_오차율(%)': [total_poly_err_pct]
                })
                
                final_res_df = pd.concat([res_display, subtotal_row], ignore_index=True)
                
                # [수정됨] 결측치(미래 실적)는 '-'로 표시하고 콤마를 찍어주는 함수
                def style_dataframe(df, key_col='일자'):
                    def highlight_row(row):
                        if row[key_col] == '[ 소계 ]':
                            return ['background-color: #e6e6e6; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    format_dict = {col: lambda x: f"{x:,.1f}" if pd.notnull(x) else "-" for col in df.columns if col != key_col}
                    return df.style.apply(highlight_row, axis=1).format(format_dict)
                
                st.markdown("#### 📅 일일 예측 결과 비교")
                st.dataframe(style_dataframe(final_res_df, '일자'), use_container_width=True, hide_index=True)
                
                # ----------------- 월별 합산 표 및 그래프 -----------------
                pred_df['월'] = pred_df['일자'].dt.strftime('%Y-%m')
                monthly_df = pred_df.groupby('월').agg({
                    '공급량 실적(GJ)': lambda x: x.sum(min_count=1),
                    'HDD_예측 공급량(GJ)': 'sum',
                    '3차 다항식 예측 공급량(GJ)': 'sum'
                }).reset_index()
                
                monthly_df['HDD_예측오차(GJ)'] = monthly_df['HDD_예측 공급량(GJ)'] - monthly_df['공급량 실적(GJ)']
                monthly_df['HDD_오차율(%)'] = (monthly_df['HDD_예측오차(GJ)'] / monthly_df['공급량 실적(GJ)']) * 100
                monthly_df['다항식_예측오차(GJ)'] = monthly_df['3차 다항식 예측 공급량(GJ)'] - monthly_df['공급량 실적(GJ)']
                monthly_df['다항식_오차율(%)'] = (monthly_df['다항식_예측오차(GJ)'] / monthly_df['공급량 실적(GJ)']) * 100
                
                # 월별 소계
                valid_monthly = monthly_df.dropna(subset=['공급량 실적(GJ)'])
                m_sum_actual = valid_monthly['공급량 실적(GJ)'].sum()
                m_sum_hdd_pred = valid_monthly['HDD_예측 공급량(GJ)'].sum()
                m_sum_poly_pred = valid_monthly['3차 다항식 예측 공급량(GJ)'].sum()

                m_subtotal = pd.DataFrame({
                    '월': ['[ 소계 ]'],
                    '공급량 실적(GJ)': [monthly_df['공급량 실적(GJ)'].sum(min_count=1)],
                    'HDD_예측 공급량(GJ)': [monthly_df['HDD_예측 공급량(GJ)'].sum()],
                    'HDD_예측오차(GJ)': [m_sum_hdd_pred - m_sum_actual if m_sum_actual > 0 else np.nan],
                    'HDD_오차율(%)': [((m_sum_hdd_pred - m_sum_actual) / m_sum_actual * 100) if m_sum_actual > 0 else np.nan],
                    '3차 다항식 예측 공급량(GJ)': [monthly_df['3차 다항식 예측 공급량(GJ)'].sum()],
                    '다항식_예측오차(GJ)': [m_sum_poly_pred - m_sum_actual if m_sum_actual > 0 else np.nan],
                    '다항식_오차율(%)': [((m_sum_poly_pred - m_sum_actual) / m_sum_actual * 100) if m_sum_actual > 0 else np.nan]
                })
                
                final_monthly_df = pd.concat([monthly_df, m_subtotal], ignore_index=True)
                
                m_cols = ['월', '공급량 실적(GJ)', 'HDD_예측 공급량(GJ)', 'HDD_예측오차(GJ)', 'HDD_오차율(%)', '3차 다항식 예측 공급량(GJ)', '다항식_예측오차(GJ)', '다항식_오차율(%)']
                final_monthly_df = final_monthly_df[m_cols]
                
                st.markdown("---")
                st.markdown("#### 📆 월별 예측 결과 비교")
                st.dataframe(style_dataframe(final_monthly_df, '월'), use_container_width=True, hide_index=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 일별 추정 결과 다운로드 (CSV)", data=final_res_df.to_csv(index=False).encode('utf-8-sig'), file_name="Daily_Estimation_Results.csv", mime="text/csv")
                with col2:
                    st.download_button("📥 월별 추정 결과 다운로드 (CSV)", data=final_monthly_df.to_csv(index=False).encode('utf-8-sig'), file_name="Monthly_Estimation_Results.csv", mime="text/csv")

                # [수정됨] 월별 그래프를 직관적인 꺾은선(line_chart)으로 변경
                st.markdown("#### 📊 월별 예측 모델 성능 비교 그래프")
                chart_data = monthly_df.set_index('월')[['공급량 실적(GJ)', 'HDD_예측 공급량(GJ)', '3차 다항식 예측 공급량(GJ)']]
                st.line_chart(chart_data)

            else:
                st.warning("예측 기간에 해당하는 기상 데이터가 구글 시트에 없습니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
