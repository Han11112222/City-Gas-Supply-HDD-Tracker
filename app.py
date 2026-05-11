import streamlit as st
import pandas as pd

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="DSE 수요예측 : HDD Dynamic Analyzer", layout="wide")
st.title("🔥 DSE 수요예측 : HDD Dynamic Analyzer")

# 2. HDD 설명 박스 (최상단)
st.info("""
### 💡 난방도일(HDD, Heating Degree Days) 계산법
난방도일은 **'기준 온도(18℃) - 일일 평균기온'**으로 계산됩니다.
- **기온이 18도보다 낮을 때:** 그 차이만큼 HDD 값이 커지며 난방 수요가 증가함을 의미합니다.
- **기온이 18도보다 높을 때:** 난방이 필요 없는 상태로 간주하여 HDD는 **0**이 됩니다.
- 공식: $HDD = \max(18 - 평균기온, 0)$
""")

# 3. 구글 시트 데이터 불러오기
@st.cache_data
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    df = pd.read_csv(sheet_url)
    return df

try:
    raw_df = load_data()
    
    # 데이터 추출 및 전처리
    df = pd.DataFrame()
    df['일자'] = pd.to_datetime(raw_df.iloc[:, 0], errors='coerce') # 날짜 형식 변환
    df['공급량(MJ)'] = raw_df.iloc[:, 1].astype(str).str.replace(',', '').astype(float)
    df['평균기온'] = pd.to_numeric(raw_df.iloc[:, 3], errors='coerce')
    
    # 4. 필터링 및 정렬 (2010년 이후 데이터만, 최신순)
    df = df.dropna(subset=['일자'])
    df = df[df['일자'].dt.year >= 2010]
    
    # HDD 및 GJ 계산
    base_temp = 18.0
    df['HDD'] = df['평균기온'].apply(lambda x: max(base_temp - x, 0) if pd.notnull(x) else 0)
    df['공급량(GJ)'] = (df['공급량(MJ)'] / 1000).round(1)
    df['평균기온'] = df['평균기온'].round(1)
    df['HDD'] = df['HDD'].round(1)

    # 화면 표시용 데이터프레임 (최신순 정렬)
    display_df = df[['일자', '공급량(GJ)', '평균기온', 'HDD']].sort_values(by='일자', ascending=False)
    # 날짜 출력 형식 변경 (YYYY-MM-DD)
    display_df['일자'] = display_df['일자'].dt.strftime('%Y-%m-%d')

    # 5. 동적 그래프 시각화
    st.subheader("📈 HDD 및 공급량 동적 추이")
    st.write("그래프를 드래그하여 특정 구간을 확대해 볼 수 있습니다.")
    
    # Streamlit 내장 line_chart는 인터랙티브 기능을 지원합니다.
    # 공급량과 HDD를 함께 보기 위해 인덱스 설정
    chart_data = df.set_index('일자')[['HDD', '공급량(GJ)']]
    st.line_chart(chart_data)

    # 6. 데이터 표 출력
    st.subheader("📊 일별 상세 데이터 (2010년 이후, 최신순)")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"데이터 처리 중 오류 발생: {e}")
