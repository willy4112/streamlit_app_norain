# 乾旱預測儀表板系統

網址：https://appappnorain-hmwcg6ukzvhjz8vcaobjx8.streamlit.app/
---
### 專案目的
* 利用每日降雨量預測乾旱的發生機率。

### 參考資料
* 中央氣象署,署屬站-台中(467490),日資料-降雨量,1897/01/01 - 2024/12/31。
* 災防中心,乾旱月資料 (https://dra.ncdr.nat.gov.tw/Frontend/Disaster/RiskDetail/BAL0000022)

### 模型建立
* 使用台中測站(467490)之降雨量資料進行訓練。
* 使用python 3.11版本，XGBoost模型訓練，並對資料進行SMOTE平衡。

### 預警條件
1. 過去6個月累積雨量<500mm。
2. 過去1年累積不降雨日>230 天。
3. 模型預測>55%。
