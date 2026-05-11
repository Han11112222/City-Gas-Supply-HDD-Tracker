import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="도시가스 HDD 추출기", layout="wide")
st.title("🔥 대성 수요예측: 일별 난방도일(HDD) 대시보드")
st.write("구글 시트의 일별 평균기온 데이터를 바탕으로 난방도일(HDD)을 자동 계산합니다.")

# 구글 시트 데이터 불러오기 (캐싱 적용으로 속도 향상)
@st.cache_data
def load_data():
    # 원본 URL의 /edit?gid=0 부분을 /export?format=csv&gid=0 으로 변경하여 CSV로 직접 읽기
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    
    # 1행이 헤더이므로 바로 읽어옵니다
    df = pd.read_csv(sheet_url)
    
    # 숫자 계산을 위해 쉼표(,) 제거 및 숫자형으로 변환
    if '공급량(MJ)' in df.columns:
        df['공급량(MJ)'] = df['공급량(MJ)'].astype(str).str.replace(',', '').astype(float)
    if '평균기온(°C)' in df.columns:
        df['평균기온(°C)'] = pd.to_numeric(df['평균기온(°C)'], errors='coerce')
        
    return df

try:
    df = load_data()
    
    # 빈 행(NaN) 제거 (일자가 없는 빈 칸 정리)
    df = df.dropna(subset=['일자'])
    
    # 1. 난방도일(HDD) 계산: MAX(18 - 평균기온, 0)
    base_temp = 18.0
    df['HDD'] = df['평균기온(°C)'].apply(lambda x: max(base_temp - x, 0) if pd.notnull(x) else 0)
    
    # 2. 공급량(MJ)을 공급량(GJ)로 스케일업 변환 (MJ / 1000 = GJ)
    if '공급량(MJ)' in df.columns:
        df['공급량(GJ)'] = df['공급량(MJ)'] / 1000
    
    # 주요 데이터만 선택하여 보여주기
    display_cols = ['일자', '공급량(GJ)', '평균기온(°C)', 'HDD']
    display_df = df[display_cols].copy()
    
    # 소수점 정리
    display_df['공급량(GJ)'] = display_df['공급량(GJ)'].round(1)
    display_df['HDD'] = display_df['HDD'].round(1)
    
    st.subheader("📊 일별 HDD 및 공급량 데이터 (GJ 기준)")
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("📈 최근 HDD 변화 추이")
    # 그래프를 그리기 위해 일자를 인덱스로 설정
    chart_data = display_df.set_index('일자')['HDD']
    st.line_chart(chart_data)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
