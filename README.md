# 🏥 仁愛醫院 - 骨壞死 (ONFH) 分期深度學習專案

## 📖 專案簡介
本專案旨在利用深度學習技術，針對**骨壞死（Osteonecrosis of the Femoral Head, ONFH）**的 X 光影像進行自動化分期（Ficat Classification Stage 1~4）。

---

## 🏗️ 系統架構
本系統採用 **階層式二元分類 (Hierarchical Binary Classification)** 策略，流程如下：

1.  **ROI 偵測 (YOLOv8)**：
    * 自動定位並裁切股骨頭區域。
2.  **分層分類**：
    * **Model 1**：區分 `Stage 4 (重症)` vs `其他`。
    * **Model 2**：區分 `Stage 1 (輕症)` vs `其他` (採用 Stacking 集成策略)。
    * **Model 3**：區分 `Stage 2` vs `Stage 3`。
3.  **邏輯整合**：
    * 綜合上述模型輸出最終分期結果。

---

## 📂 檔案結構說明

| 檔案/資料夾 | 說明 |
| :--- | :--- |
| `01_yolo_train.ipynb` | 訓練 YOLO 模型以定位病灶。 |
| `02_yolo_infer_make_ROI.ipynb` | 使用 YOLO 裁切出 ROI (Region of Interest) 作為分類器輸入。 |
| `03_stage_classifier...` | 訓練各個階段的二元分類器 (M1, M2, M3)。 |
| `03.25_stage_classifier_m2_stacking.ipynb` | 針對 M2 模型進行 Stacking 集成學習訓練。 |
| `03.5_stage_classifier_combinationV9.ipynb` | 整合所有模型進行最終推論與評估。 |
| `04_feature_spaceV2.ipynb` | 繪製 t-SNE 圖以分析特徵空間分布。 |
| `.gitignore` | 設定 Git 忽略檔案清單（權重、數據集、暫存檔）。 |

---

## ⚙️ 環境與資料設定
> ⚠️ **注意**：由於病患隱私與檔案大小限制，本倉庫 **不包含** 原始影像數據與訓練好的模型權重 (`.pt`, `.pth`)。

### 🚀 如何開始
1.  請將原始 Labelme 標註資料放入 `drive-download-xxxx` 資料夾。
2.  依序執行 Notebook `01` -> `02` 產生訓練數據。
3.  執行 `03` 系列 Notebook 進行模型訓練。

---