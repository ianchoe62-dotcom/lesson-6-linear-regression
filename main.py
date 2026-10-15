import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

# 1. 데이터 로드 및 전처리 캐싱
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 형식 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 2025년까지만 필터링
    df = df[df["연도"] <= 2025]
    
    # 연도별 데이터 수(관측일 수)와 평균기온 계산
    yearly_stats = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 관측일이 300일 이상인 해만 남기기
    filtered_yearly = yearly_stats[yearly_stats["관측일수"] >= 300].copy()
    
    return filtered_yearly

# 데이터 불러오기
try:
    df_yearly = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 회귀 분석 모델 학습
X = df_yearly[["연도"]]
y = df_yearly["연평균기온"]

model = LinearRegression()
model.fit(X, y)

# 상관계수 계산
correlation = df_yearly["연도"].corr(df_yearly["연평균기온"])

# 회귀선 데이터 생성
df_yearly["예측기온"] = model.predict(X)

# --- UI 구성 ---
st.title("🌡️ 서울 연도별 평균기온 예측기")
st.markdown("서울의 기온 데이터를 바탕으로 연도별 연평균기온의 추세를 분석하고 미래 기온을 예측합니다.")

# 학습 정보 요약
n_years = len(df_yearly)
start_year = int(df_yearly["연도"].min())
end_year = int(df_yearly["연도"].max())

st.info(
    f"📊 **학습 정보** | "
    f"사용한 해의 개수: **{n_years}개** | "
    f"시작 연도: **{start_year}년** | "
    f"끝 연도: **{end_year}년** (관측일 300일 이상, 2025년까지 기준)"
)

# 레이아웃 나누기 (왼쪽: 그래프, 오른쪽: 예측 슬라이더)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 연도별 연평균기온 및 회귀선")
    
    # Plotly 그래프 생성
    fig = px.scatter(
        df_yearly, x="연도", y="연평균기온",
        labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
        title=f"서울 연평균기온 산점도 및 추세선 (상관계수: {correlation:.2f})"
    )
    
    # 산점도 마커 스타일 지정
    fig.update_traces(marker=dict(size=8, color="royalblue"), name="실제 연평균기온")
    
    # 회귀선 추가
    fig.add_scatter(
        x=df_yearly["연도"], 
        y=df_yearly["예측기온"], 
        mode="lines", 
        name="회귀 직선",
        line=dict(color="red", width=2)
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔮 연도별 기온 예측")
    st.markdown("슬라이더를 움직여 특정 연도의 예상 평균기온을 확인하세요.")
    
    # 슬라이더 (1900 ~ 2100)
    selected_year = st.slider(
        "조회할 연도 선택", 
        min_value=1900, 
        max_value=2100, 
        value=2026, 
        step=1
    )
    
    # 선택한 연도의 기온 예측
    predicted_temp = model.predict([[selected_year]])[0]
    
    # 결과 강조 표시
    st.markdown("---")
    st.markdown(f"### **{selected_year}년** 예상 연평균기온")
    st.markdown(
        f"<h1 style='color: #ff4b4b; text-align: center;'>{predicted_temp:.2f} °C</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    if selected_year > 2025:
        st.caption("💡 2025년 이후의 연도는 기존 기온 상승 추세를 바탕으로 한 단순 예측값입니다.")
    elif selected_year < start_year:
        st.caption("💡 데이터 수집 시작 이전 연도의 추정값입니다.")
    else:
        # 실제 데이터가 있는 경우 비교값 제공
        actual_row = df_yearly[df_yearly["연도"] == selected_year]
        if not actual_row.empty:
            actual_temp = actual_row["연평균기온"].values[0]
            st.info(f"실제 관측된 연평균기온: **{actual_temp:.2f} °C**")
