# 🏥 仁愛醫院 - 骨壞死 (ONFH) 分期深度學習專案

## 📖 專案簡介
本專案旨在利用深度學習技術，針對**骨壞死（Osteonecrosis of the Femoral Head, ONFH）**的 X 光影像進行自動化分期（Ficat Classification Stage 1~4）。

---

## 🏗️ 系統架構
本系統採用 **階層式二元分類 (Hierarchical Binary Classification)** 策略，流程如下：

1.  **ROI 偵測 (YOLOv8)**：
    * 自動定位並裁切股骨頭區域。
2.  **分層分類**：
    * **Model 1**：區分 `Stage 4 (重症)` vs `其他` (使用 **DenseNet121**)。
    * **Model 2**：區分 `Stage 1 (輕症)` vs `其他` (基於 **EfficientNet-B0, ResNet50, ConvNeXt-Tiny** 採用 Stacking 集成策略)。
    * **Model 3**：區分 `Stage 2` vs `Stage 3` (使用 **EfficientNet-B0**)。
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

> ⚠️ **注意**：由於病患隱私與檔案大小限制，本倉庫 **不包含** 原始影像數據。
> 💾 **模型權重下載**：訓練好的模型權重檔 (`.pt`, `.pkl`) 已上傳至 **GitHub Releases**。請至 Repo 右側的 **Releases** 區塊下載所需的權重檔案，並放置於專案的工作目錄（或 `weights/` 資料夾）中。

### 🔧 環境安裝 (Installation)

本專案建議使用 Anaconda 進行環境管理。請選擇以下其中一種方式建立環境：

**方法一：使用 environment.yaml (推薦，最完整)**
這會完美複製開發者的 Conda 環境配置 (包含 CUDA 等依賴)。
```bash
# 建立環境
conda env create -f environment.yaml

# 啟用環境 (請依據 yaml 內的 name，通常是 unet_labeling)
conda activate unet_labeling
```

**方法二：使用 requirements.txt**
如果您不使用 Conda，可以使用 pip 安裝核心套件。
```bash
pip install -r requirements.txt
```

### 🚀 如何開始
1.  請將原始 Labelme 標註資料放入 `drive-download-xxxx` 資料夾。
2.  依序執行 Notebook `01` -> `02` 產生訓練數據。
3.  執行 `03` 系列 Notebook 進行模型訓練。

---

## 🛠️ 開發者指南：如何更新與上傳版本

這部分是給開發者的 Git 操作備忘錄。

### 1. 確認目前狀態
```bash
git status
```

### 2. 加入修改 (Add)
```bash
git add .
```

### 3. 提交版本 (Commit)
```bash
git commit -m "在此寫下更新說明"
```

### 4. 推送到雲端 (Push)
```bash
git push
```