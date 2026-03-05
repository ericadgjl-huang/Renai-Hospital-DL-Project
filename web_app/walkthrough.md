# 開發完成：髖關節嚴重程度預測分類網頁

本開發已完成使用者指定的網頁應用設計，並且完全放置在一個獨立的專案資料夾內，不會影響原始外部程式的運行與結構。

## 修改與新增項目

- **[NEW] `web_app/` 資料夾**:
  - 用於隔離網頁程式與模型載入代碼，保持專案整潔。
- **[NEW] `web_app/app.py`**:
  - Flask 背景伺服器程式碼。
  - 使用 `02_yolo_infer_make_ROI.ipynb` 相同的邏輯進行 YOLO 髖部裁切（左 L 則取 `x2` 最大者，右 R 取 `x1` 最小者）。
  - 使用 `03.5_stage_classifier_combinationV9.ipynb` 相同的決策樹與模型組合架構：
    - m1 (DenseNet121): 判別 Stage 4。
    - m2 (多模型 Stacking): 判別 Stage 1。
    - m3 (DenseNet121): 判別 Stage 3 vs Stage 2。
  - 以 JSON API 的形式回傳裁切後的預覽圖（Base64編碼）、最終分類階段及各模型信心水準。
- **[NEW] `web_app/templates/index.html`**:
  - 現代化且動態互動的網頁前端（基於 Tailwind CSS 設計）。
  - 提供影像選擇與上傳、左/右判定勾選。
  - 於左側提交後，右側會顯示：
    1. **「YOLO 裁切目標區域」** (移至左下角)
    2. **「AI 關注區域 (Grad-CAM 熱力圖)」** (包含 m1, m2, m3 三個獨立模型的熱力圖並排顯示)
    3. **「最終判定分類 (第 1-4 類)」**
    4. **「各類綜合判定機率排名」**（將三階段模型機率合併計算出各類別最終機率，並由高至低自動排序）。
       * ※ **機率換算圖解邏輯：**
         * 模型 `m1` 判定是第 4 類的機率為 `P(Stage4)`。因此不是第 4 類的機率為 `P(Not_4) = 1 - P(Stage4)`
         * 模型 `m2` 判定是第 1 類的機率為 `P(Stage1_nodal)`。因此不是第 1 類的機率為 `1 - P(Stage1_nodal)`
         * 模型 `m3` 判定是第 3 類的機率為 `P(Stage3_nodal)`。因此不是第 3 類的機率為 `1 - P(Stage3_nodal)`
       * 故網頁最終呈現的四類絕對機率為：
         * **第 4 類機率** = `P(Stage4)`
         * **第 1 類機率** = `P(Not_4) × P(Stage1_nodal)`
         * **第 3 類機率** = `P(Not_4) × (1 - P(Stage1_nodal)) × P(Stage3_nodal)`
         * **第 2 類機率** = `P(Not_4) × (1 - P(Stage1_nodal)) × (1 - P(Stage3_nodal))`

## 如何在本地執行

1. 打開終端機或命令提示字元 (Command Prompt/PowerShell)。
2. 進入到剛剛新增的 `web_app` 資料夾：
   ```bash
   cd  /d "D:\中興大學\碩一上\仁愛醫院\仁愛醫院調整8總整理\web_app"
   ```
3. 執行 Flask 伺服器：
   ```bash
   python app.py
   ```
   *注意：會先在主控台印出 `Loading Models...`，需要稍待幾秒鐘直到模型載入完成（出現 `Models Initialized Successfully!`）*
4. 在瀏覽器打開以下網址：
   [http://127.0.0.1:5000](http://127.0.0.1:5000)
5. 在美觀的圖形介面中上傳影像，選擇左右側後進行分析。

## 視覺化設計特色
1. **動態載入**：在伺服器計算裁切與神經網路推論時，會有明確的「載入中 (Loading...)」設計提示。
2. **錯誤處理**：如果 YOLO 模型抓不到髖關節、沒有預測框，會以明確紅字回覆錯誤原因，確保程式不會輕易當機。
3. **優美與響應性**：使用了毛玻璃背景特效 (Glassmorphism)，配合圓角框與柔和字體，視覺體驗佳。

如有需要調整信心閾值 (Threshold)，亦可直接於 `app.py` 中的 `thr2_val` 或 Node 判斷內進行微調。
