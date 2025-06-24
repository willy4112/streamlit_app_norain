import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# st.set_page_config(page_title="乾旱預測儀表板", layout="wide")
st.set_page_config(page_title="乾旱預測儀表板")

# 頁籤切換
tab1, tab2, tab3 = st.tabs(["📁 上傳資料", "📊 查看預測", "🧠 解釋模型"])

with tab1:
    st.header("📂 上傳模型與資料檔案")

    # 分為兩欄顯示
    col1, col2 = st.columns(2)

    # 左欄：上傳模型
    with col1:
        st.subheader("1. 請上傳模型檔案")
        model_file = st.file_uploader("選擇 XGBoost模型檔案（.pkl）", type=["pkl"], key="model")

        if model_file is None:
            st.warning("請先上傳模型檔案後再進行預測。")
            st.stop()

        try:
            model = joblib.load(model_file)
            st.success("✅ 模型載入成功")
        except Exception as e:
            st.error(f"❌ 模型載入失敗：{e}")
            st.stop()

    # 右欄：上傳資料與欄位選擇
    with col2:
        st.subheader("2. 請上傳氣象資料")
        uploaded_file = st.file_uploader("選擇氣象資料（.csv 或 .xlsx）", type=["csv", "xlsx"], key="data")

        if not uploaded_file:
            st.warning("請上傳含 365 筆資料的氣象檔案（CSV 或 Excel）")
            st.stop()
        else:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if len(df) < 365:
                    st.error("資料時間長度不夠，需 ≥ 365 筆。")
                    st.stop()

                st.success("✅ 資料檔案上傳成功")
            except Exception as e:
                st.error(f"❌ 資料讀取失敗或格式錯誤：{e}")
                st.stop()

    # 3. 選擇欄位
    st.subheader("3. 選擇資料欄位")
    col1, col2 = st.columns(2)
    with col1:
        time_col = st.selectbox("📅 請選擇時間欄位", df.columns)
    with col2:
        rain_col = st.selectbox("🌧️ 請選擇降雨量欄位", df.columns)

    # 顯示所選欄位的前 10 筆資料
    st.write("前 5 筆資料")
    st.dataframe(df.head(5))

    # 5. 檢查並補齊缺值
    missing = df[df[rain_col].isna()]
    if not missing.empty:
        st.warning(f"發現 {len(missing)} 筆缺失值，進行線性插補")
        df[rain_col] = df[rain_col].interpolate(method="linear")
        filled_values = df.loc[missing.index]
        st.write("缺失值補齊如下：")
        st.dataframe(filled_values)

    # 6. 建立特徵參數
    def generate_features_monthly(df, time_col, rain_col):
        df = df.copy()
        df['date'] = df[time_col]
        df['rain'] = df[rain_col]
    
        # 需要有 year 和 month 欄位
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isnull().any():
            raise ValueError("資料中缺少正確格式的日期欄位（需要可轉換成 datetime）")
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
        # 極端降雨
        df['rain50'] = (df['rain'] >= 50).astype(int)
        df['rain0'] = (df['rain'] == 0).astype(int)
    
        # 月彙總
        df_month = df.groupby(['year', 'month']).agg({
            'rain': 'sum',
            'rain50': 'sum',
            'rain0': 'sum'
        }).reset_index()
    
        # 累積雨量與標準差
        df_month['sum_2m'] = df_month['rain'].rolling(window=2, min_periods=2).sum()
        df_month['sum_3m'] = df_month['rain'].rolling(window=3, min_periods=3).sum()
        df_month['sum_6m'] = df_month['rain'].rolling(window=6, min_periods=6).sum()
        df_month['sum_1y'] = df_month['rain'].rolling(window=12, min_periods=12).sum()
    
        df_month['std_2m'] = df_month['rain'].rolling(window=2, min_periods=2).std()
        df_month['std_3m'] = df_month['rain'].rolling(window=3, min_periods=3).std()
        df_month['std_6m'] = df_month['rain'].rolling(window=6, min_periods=6).std()
        df_month['std_1y'] = df_month['rain'].rolling(window=12, min_periods=12).std()
    
        # 0降雨累積與標準差
        df_month['rain0_sum_3m'] = df_month['rain0'].rolling(window=3, min_periods=3).sum()
        df_month['rain0_sum_6m'] = df_month['rain0'].rolling(window=6, min_periods=6).sum()
        df_month['rain0_sum_1y'] = df_month['rain0'].rolling(window=12, min_periods=12).sum()
    
        df_month['rain0_std_3m'] = df_month['rain0'].rolling(window=3, min_periods=3).std()
        df_month['rain0_std_6m'] = df_month['rain0'].rolling(window=6, min_periods=6).std()
        df_month['rain0_std_1y'] = df_month['rain0'].rolling(window=12, min_periods=12).std()
    
        # 每月平均雨量差異
        month_avg = df_month.groupby('month')['rain'].mean().reset_index()
        month_avg.rename(columns={'rain': 'rain_avg'}, inplace=True)
        df_month = df_month.merge(month_avg, on='month', how='left')
        df_month['rain_deviation'] = df_month['rain'] - df_month['rain_avg']
    
        # 移除不必要欄位
        df_month = df_month.drop(['rain_avg', 'year', 'month'], axis=1)
    
        return df_month.dropna().reset_index(drop=True)
    features = generate_features_monthly(df, time_col, rain_col)


# 頁籤2：查看預測結果
with tab2:
    st.header("📊 預測結果與趨勢")

    try:
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(features)
            predictions = model.predict(dmatrix)
    
        elif isinstance(model, xgb.XGBRegressor):
            predictions = model.predict(features)
    
        elif isinstance(model, xgb.XGBClassifier):
            predictions = model.predict(features)
            proba = model.predict_proba(features)
    
            st.subheader("預測機率")
            proba_df = pd.DataFrame(proba, columns=[f"Class_{i}" for i in range(proba.shape[1])])
            # 顯示預測機率（最後一筆）
            last_index = -1
            last_prob = proba_df['Class_1'].iloc[last_index]
            last_time = df[time_col].iloc[last_index]  # 或者從原始資料抓取對應時間欄位
            
            # 建立兩欄佈局
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"時間：\t{last_time}")
            
            with col2:
                if last_prob < 0.45:
                    st.metric(label="🟢 綠燈", value=f"{last_prob:.2%}", delta="機率 < 45%", delta_color="off")
                elif last_prob <= 0.55:
                    st.metric(label="🟡 黃燈", value=f"{last_prob:.2%}", delta="機率 45%～55%", delta_color="off")
                else:
                    st.metric(label="🔴 紅燈", value=f"{last_prob:.2%}", delta="機率 > 55%", delta_color="off")
    
        else:
            st.error("模型類型不支援，請確認為 Booster、XGBRegressor 或 XGBClassifier")
            st.stop()
    
    except Exception as e:
        st.error(f"預測錯誤：{e}")
        st.stop()
    
    # 8. 繪製雨量折線圖
    try:
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.sans-serif'] = 'SimHei'
        plt.rcParams["axes.unicode_minus"] = False
        
        # 取出最後一筆資料
        show = features.iloc[-1:, :]
        
        # 取出最近 6 個月累積雨量
        total_rainfall_180 = show['sum_6m'].iloc[0]
        
        # 確保時間欄位是 datetime 格式
        df[time_col] = pd.to_datetime(df[time_col])
        
        # 時間排序後取出最後 180 筆，計算起訖時間
        df_sorted = df.sort_values(by=time_col)
        start_date = df_sorted[time_col].iloc[-180]
        end_date = df_sorted[time_col].iloc[-1]
        
        st.subheader("結果說明")
        # 使用兩欄，左邊放文字，右邊放紅綠燈
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.info(f"過去 180 天（{start_date.date()} - {end_date.date()}）累積降雨量為：{total_rainfall_180:.1f} mm")
        
        with col2:
            if total_rainfall_180 < 500:
                st.error("🔴 紅燈")
            else:
                st.success("🟢 綠燈")
        
        # 1年累積不降雨日數
        rain0_sum_1y = show['rain0_sum_1y'].iloc[0]
        # 使用兩欄，左邊放文字，右邊放紅綠燈
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.info(f"過去 1年 累積不降雨日數為：{rain0_sum_1y:.0f} 天")
        
        with col2:
            if rain0_sum_1y > 230:
                st.error("🔴 紅燈")
            else:
                st.success("🟢 綠燈")
        
        
        # 判斷第一層：rain0_sum_1y
        if rain0_sum_1y > 230:
            # 符合第一層條件 → 綠燈（繼續判斷第二層）
            if total_rainfall_180 < 500:
                # 符合第二層條件 → 黃燈（進入第三層）
                if last_prob > 0.55:
                    st.error("🔴 警示燈號：紅，開始行動")
                else:
                    st.warning("🟡 警示燈號：黃，準備行動")
            else:
                # 不符合第二層條件 → 綠燈
                st.success("🟢 警示燈號：綠，需要注意")
        else:
            # 不符合第一層條件 → 灰燈
            st.info("⚪ 警示燈號：灰，無")
        
        
        
        
        st.subheader("")
        # 繪製折線圖
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(df[time_col], df[rain_col], linewidth=2)
        ax.set_xlabel("")
        ax.set_ylabel("降雨量 (mm)")
        ax.set_title("每日降雨量變化趨勢")
        ax.grid(False)
        
        st.pyplot(fig)
        

    
    except Exception as e:
        st.error(f"繪圖失敗：{e}")

# 頁籤3：解釋模型
with tab3:
    # st.header("🧠 模型說明")

    st.markdown("### 模型建立")
    st.markdown("- 使用`台中測站(467490)`之降雨量資料進行訓練。")
    st.markdown("- 使用`python 3.11`版本，`XGBoost`模型訓練，並對資料進行`SMOTE`平衡。")
    st.markdown("### 重要特徵")
    # 特徵重要性資料
    importance_data = {
        "Feature": [
            "rain0_sum_1y", "rain50", "rain0_std_1y", "sum_1y", "sum_6m",
            "rain_deviation", "rain0_sum_6m", "std_1y", "std_2m", "rain0_std_6m"
        ],
        "Importance": [
            0.1481069, 0.11731146, 0.06896101, 0.060764737, 0.059234254,
            0.056553446, 0.054832846, 0.05454356, 0.04779761, 0.046084266
        ]
    }
    df_importance = pd.DataFrame(importance_data).sort_values("Importance", ascending=False)
    # 畫圖
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(df_importance["Feature"], df_importance["Importance"])
    ax.set_xlabel("Feature Importance")
    # ax.set_title("特徵重要性 (Top 10)")
    ax.invert_yaxis()  # 讓重要性高的在上面

    st.pyplot(fig)
    
    st.markdown("- 首先標記不降雨日(`rain0`)與強降雨(`rain50`)，接著標記月份，將日資料轉換為越累積資料。")
    st.markdown("- `rain0_sum_1y`：過去一年累積的不降雨日數。")
    st.markdown("- `rain50`：月累積降雨 >= 50mm 的日數。")
    st.markdown("- `rain0_std_1y`：過去一年月累積不降雨日數的標準差。")


    st.markdown("### 預警條件")
    st.markdown("""
        - 過去6個月累積雨量<500mm。
        - 過去1年累積不降雨日>230 天。
        - 模型預測>55%。
    """)

