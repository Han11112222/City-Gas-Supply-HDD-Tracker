import streamlit as st
import pandas as pd

# 1. 페이지 설정 및 제목 (DSE 수요예측 포맷 적용)
st.set_page_config(page_title="DSE 수요예측 : HDD Analyzer", layout="wide")
st.title("🔥 DSE 수요예측 : HDD Analyzer")
st.write("구글 시트의 일별 평균기온 데이터를 바탕으로 난방도일(HDD)을 자동 계산합니다.")

# 2. 구글 시트 데이터 불러오기
@st.cache_data
def load_data():
    # Han형님께서 제공해주신 raw 데이터 링크
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    df = pd.read_csv(sheet_url)
    return df

try:
    raw_df = load_data()
    
    # 3. 에러 방지용 데이터 추출 (이름 대신 열의 위치 번호로 추출)
    # 파이썬은 0부터 숫자를 세므로 A열=0, B열=1, C열=2, D열=3 입니다.
    df = pd.DataFrame()
    df['일자'] = raw_df.iloc[:, 0]        # A열: 일자
    df['공급량(MJ)'] = raw_df.iloc[:, 1]  # B열: 공급량(MJ)
    df['평균기온'] = raw_df.iloc[:, 3]    # D열: 평균기온
    
    # 4. 데이터 전처리
    # 일자가 비어있는 불필요한 행 제거
    df = df.dropna(subset=['일자'])
    
    # 숫자 계산을 위해 쉼표(,) 제거 및 숫자형으로 확실하게 변환
    df['공급량(MJ)'] = df['공급량(MJ)'].astype(str).str.replace(',', '').astype(float)
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    
    # 5. 난방도일(HDD) 및 GJ 스케일업 계산
    base_temp = 18.0
    # 평균기온 값이 정상적인 숫자일 때만 HDD 계산: MAX(18 - 평균기온, 0)
    df['HDD'] = df['평균기온'].apply(lambda x: max(base_temp - x, 0) if pd.notnull(x) else 0)
    
    # MJ를 GJ로 변환
    df['공급량(GJ)'] = df['공급량(MJ)'] / 1000
    
    # 6. 화면 출력용 데이터 정리
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD']].copy()
    display_df['공급량(GJ)'] = display_df['공급량(GJ)'].round(1)
    display_df['HDD'] = display_df['HDD'].round(1)
    
    # 7. Streamlit 화면에 시각화
    st.subheader("📊 일별 HDD 및 공급량 데이터 (GJ 기준)")
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("📈 최근 HDD 변화 추이")
    chart_data = display_df.set_index('일자')['HDD']
    st.line_chart(chart_data)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
    st.write("컬럼 인덱스 매핑에 문제가 있을 수 있습니다. 시트의 A열이 일자, D열이 평균기온인지 확인해주세요.")
