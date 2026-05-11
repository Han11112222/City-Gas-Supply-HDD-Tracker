import streamlit as st
import pandas as pd

# 1. 페이지 설정
st.set_page_config(page_title="DSE Demand Forecast: HDD Analyzer", layout="wide")
st.title("🔥 DSE Demand Forecast: HDD Analyzer")

st.info("""
### 💡 난방도일(HDD, Heating Degree Days) 계산법
난방도일은 실외 온도가 기준 온도보다 낮아질 때 에너지 소비가 발생하는 원리를 이용한 지수입니다.
- **공식:** $HDD = \max(18.0 - \text{평균기온}, 0)$
- **의미:** 기온이 18℃보다 낮을 때 그 차이만큼 HDD 값이 생성되며, 이 값이 클수록 난방 수요가 증가합니다. 18℃ 이상일 경우 HDD는 0이 됩니다.
""")

# 2. 핵심 최적화: 데이터 로드 및 '전처리 과정 전체'를 캐싱합니다.
@st.cache_data
def load_and_process_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    raw_df = pd.read_csv(sheet_url)
    
    df = pd.DataFrame()
    # 속도 개선 포인트: format='%Y-%m-%d' 추가로 날짜 추론 연산 시간 제거
    df['일자'] = pd.to_datetime(raw_df.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
    df['공급량(MJ)'] = raw_df.iloc[:, 1].astype(str).str.replace(',', '').astype(float)
    df['평균기온'] = pd.to_numeric(raw_df.iloc[:, 3], errors='coerce')
    
    # 데이터 필터링 (결측치 제거 및 2010년 이후)
    df = df.dropna(subset=['일자', '평균기온'])
    df = df[df['일자'].dt.year >= 2010]
    
    # HDD 및 단위 변환 일괄 계산
    base_temp = 18.0
    df['HDD'] = df['평균기온'].apply(lambda x: max(base_temp - x, 0) if pd.notnull(x) else 0)
    df['공급량(GJ)'] = (df['공급량(MJ)'] / 1000).round(1)
    df['평균기온'] = df['평균기온'].round(1)
    df['HDD'] = df['HDD'].round(1)
    
    return df

try:
    # 이제 무거운 작업은 최초 1회만 실행되고, 이후에는 메모리에서 즉각 불러옵니다.
    df = load_and_process_data()

    # 화면에 표시할 데이터프레임 복사 및 정렬 (최신순)
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD']].sort_values(by='일자', ascending=False).copy()
    display_df['일자'] = display_df['일자'].dt.strftime('%Y-%m-%d')

    # 3. 표 출력 및 다운로드 버튼
    st.subheader("📊 일별 상세 데이터 (2010년 이후, 최신순)")
    
    csv = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 HDD 데이터 다운로드 (CSV)",
        data=csv,
        file_name="DSE_HDD_Data.csv",
        mime="text/csv",
    )
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 4. 하단 동적 그래프 출력
    st.subheader("📈 HDD 및 공급량 동적 추이")
    st.write("마우스를 올려 값을 확인하거나 드래그하여 확대할 수 있습니다.")
    chart_data = df.sort_values(by='일자').set_index('일자')[['HDD', '공급량(GJ)']]
    st.line_chart(chart_data)

except Exception as e:
    st.error(f"데이터 처리 중 오류 발생: {e}")
