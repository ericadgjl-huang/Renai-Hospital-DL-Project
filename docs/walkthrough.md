# Walkthrough — 從資料到 Demo 網頁

本文件說明：整體流程、資料怎麼切（train / val / test / cross-validation）、`run_all.bat`
每個功能、輸出檔怎麼看，以及怎麼啟動並使用本機端的 demo 網頁。

> 前提：先啟用環境 `conda activate unet_labeling`，並在專案根目錄執行。

---

## 1. 整體流程一張圖

```
drive-download-.../  (原始 X 光, 4 個 stage 資料夾)
  │  [01_prepare_stage_dataset.py]  YOLO 偵測髖關節 ROI → 裁切 → 右髖翻成左髖方向
  ▼
stage_cls_dataset/stage_1..4/  +  roi_all.csv      (287 張, 4 類)
  │  [固定切分]  outputs/splits/outer_split.json    (seed 42, 80/20)
  ▼
train_val (229)  ─────────────────────────────►  test (58)  ※ 全程不參與任何選拔
  │  [04_train_all_cuts.py]  10 cuts × 5 folds × 7 backbones
  ▼
outputs/cuts/<cut>/cv/fold_*/<backbone>/best_*.pth  (每 cut 35 個 checkpoint)
  │  [05_build_ensemble.py]  每 fold 選最佳 backbone(共 5 個) → voting vs stacking
  ▼
outputs/cuts/<cut>/ensemble/winner.json
  │  [06_search_hierarchy.py]  5 種 topology, 用 OOF 選最佳, test 只回報
  ▼
outputs/hierarchy/best_topology.json   +  search_results.csv
  │  [07_update_web.py]   寫 web_app/_runtime.json
  │  [08_report_ci.py]    test 指標的 95% 信賴區間 → outputs/report/ci_report.md
  │  [09_train_combiner.py]  學習型 combiner vs 硬路由 → outputs/combiner/comparison.csv
  ▼
web_app/app.py   本機 demo 網頁 (上傳 X 光 → Stage + Grad-CAM)
```

---

## 2. 資料怎麼切（重點）

總共 **287 張**影像（stage1=49, stage2=112, stage3=60, stage4=66）。

### 2.1 外層 80/20 切分（凍結、不可變）
- 在 `outputs/splits/outer_split.json`，用 seed 42 做 **stratified（依 4 類比例）** 切分，且**寫檔後固定**，每次重跑都沿用同一份。
- **train_val = 229 張**；**test = 58 張**（各類約 stage1=10, stage2=23, stage3=12, stage4=13）。
- **test 是「期末考卷」**：訓練、選 backbone、選 topology、選 combiner 模型，**全程都不准看 test**，只有最後回報分數時才用一次。

### 2.2 train_val 內部的 5-fold cross-validation
- 把 229 張再用 **StratifiedKFold(5)** 分成 5 折。
- 每一折輪流當 **驗證集 (val ≈ 46 張)**，其餘 4 折當 **訓練集 (train ≈ 183 張)**。
- 跑 5 次，所以**每張 train_val 影像剛好當過 1 次 val、4 次 train**。

### 2.3 「三個集合」各自的角色
| 集合 | 數量 | 角色 |
| --- | --- | --- |
| 訓練集 train | 每折 ≈183 | 用來更新 CNN 權重（class-weighted CE + AdamW + early stopping） |
| 驗證集 val | 每折 ≈46 | 決定最佳 epoch（early stop）、選該 fold 最佳 backbone、產生 **OOF** 預測 |
| 測試集 test | 58 | 最後回報；**不參與任何選拔** |

### 2.4 每個 cut 還會「過濾」
每個 binary cut 只用相關的 stage。例如 `2_vs_3` 只用 stage2+stage3（172 張），所以它的每折訓練量更少（≈110 train / ≈27 val）。這是為什麼有些 cut 特別不穩。

### 2.5 OOF（out-of-fold）是什麼、為什麼重要
某張 train_val 影像若落在第 k 折的 val，它的 OOF 預測就只由「第 k 折那個沒看過它的模型」產生。
→ OOF 是「模型沒背過答案」時的表現，**所以拿來做選拔（選 topology、選 combiner 模型）不會洩漏 test**。

### 2.6 病人層級切分（尚未啟用）⚠️
目前檔名 `S<stage>_<side><n>.jpg` **沒有病人 ID**，無法保證「同一病人不會同時出現在 train 和 test」。
程式已預留 `groups` 參數與 `patient_groups()`：只要 `roi_all.csv` 多一個 `patient` 欄位，
就會自動改用 `StratifiedGroupKFold`。**建議向醫院索取 filename→病人 對照表。**

---

## 3. `run_all.bat` 每個功能

先 `conda activate unet_labeling`，再執行。`.bat` 內容刻意全英文（cmd.exe 用 Big5 讀檔，中文會亂碼）。

| 指令 | 跑哪些步驟 | 用途 / 耗時 |
| --- | --- | --- |
| `run_all.bat`            | 3→4→5→6→7 | **預設**：用既有 checkpoints 重建全部報表。**不重訓**。數分鐘 |
| `run_all.bat rebuild`    | 3→4→5→6→7 | 同上 |
| `run_all.bat train`      | 2→3→4→5→6→7 | **重新訓練**全部模型（吃進新的正則化）再重建。**數小時** |
| `run_all.bat full`       | 1→2→3→4→5→6→7 | 從 Drive 原圖整套跑（含 YOLO 裁切）。最久 |
| `run_all.bat smoke`      | 單一冒煙測試 | 確認環境沒壞（會覆寫 3_vs_4 fold0 的一格 ckpt，僅測試用） |
| `run_all.bat web`        | 啟動 Flask | 開本機 demo 網頁 |

各步驟對應的 script：

| 步驟 | Script | 做什麼 |
| --- | --- | --- |
| 1 | `01_prepare_stage_dataset.py` | YOLO 裁 ROI、右髖翻左、輸出 `stage_cls_dataset/` 與 `roi_all.csv` |
| 2 | `04_train_all_cuts.py` | 10 cuts × 5 folds × 7 backbones 訓練（最久） |
| 3 | `05_build_ensemble.py` | 每 cut 選最佳 backbone/fold、比較 voting vs stacking |
| 4 | `06_search_hierarchy.py` | 5 種 topology，**用 OOF 選最佳**，test 只回報 |
| 5 | `07_update_web.py` | 產生 `web_app/_runtime.json` 給網頁用 |
| 6 | `08_report_ci.py` | test 指標的 bootstrap 95% 信賴區間 |
| 7 | `09_train_combiner.py` | 學習型 combiner，與硬路由 hierarchy 比較 |

> 開始前 `.bat` 會先檢查 torch 與 GPU（`[env] torch ... | device = cuda`）。若沒啟用環境會直接報錯並提示。

---

## 4. 主要輸出檔怎麼看

| 檔案 | 看什麼 |
| --- | --- |
| `outputs/ensemble_summary.csv` | 每個 binary cut 的 OOF / test 分數、各 fold 用了哪個 backbone |
| `outputs/hierarchy/search_results.csv` | 5 種 topology 的 **oof_macro_f1**（選拔依據）與 **test_macro_f1**（回報），`selected=True` 是最終選中的 |
| `outputs/hierarchy/best_topology.json` | 最終 topology、`selected_by: oof_macro_f1`、oof 與 test 分數 |
| `outputs/report/ci_report.md` | **最終 4 類 macro-F1 與 95% 信賴區間**（論文要引這個），含人類 κ 基準 |
| `outputs/combiner/comparison.csv` | combiner（學習型）vs hierarchy（硬路由）的 test 比較 |
| `web_app/_runtime.json` | 網頁啟動時讀的設定（topology + 每 cut 的 checkpoint 路徑） |

---

## 5. 本機端 Demo 網頁 walkthrough

### 5.1 啟動
```powershell
conda activate unet_labeling
run_all.bat web
```
或直接：
```powershell
python web_app/app.py
```
啟動後終端機會印：
```
[boot] topology = T1 ((1,2),(3,4))
[boot] cuts: ['12_vs_34', '1_vs_2', '3_vs_4']
```
代表它讀到 `_runtime.json` 並載入了該 topology 的 3 個 cut 模型。

> 若看到 `Missing _runtime.json` → 先跑 `run_all.bat`（或至少 `python scripts/07_update_web.py`）。
> 若改了 topology／重跑了 pipeline，要**重新啟動 Flask** 才會吃到新的 `_runtime.json`。

### 5.2 開啟網頁
瀏覽器open：
```
http://127.0.0.1:5000
```

### 5.3 操作步驟
1. **上傳影像**：點選檔案，選一張髖部 X 光（整張原圖即可，不用自己裁）。
2. **選擇左右側**：
   - `左側 (L)`（預設）／`右側 (R)`。
   - 提示：通常**畫面右邊是病人的左髖、畫面左邊是右髖**。選錯會裁到另一側。
3. 按 **「開始分析預測」**。

### 5.4 背後發生什麼（對應你的 pipeline）
```
原圖 → YOLO 偵測髖關節 → 依你選的左/右挑出對應的框 → 裁切
     → 若選右側則水平翻轉(統一成左髖方向) → Resize 384 + Normalize
     → 依 best_topology 的 3 個 cut 各做 soft-vote 機率
     → 沿 topology 樹把機率乘到 4 個 stage → 取最大者為最終 Stage
     → 每個 cut 產生 Grad-CAM 熱區
```

### 5.5 結果畫面會看到
- **最終預測類別**：`Stage 1/2/3/4`。
- **各類別機率**：第1~4類的機率排序（這是沿 topology 樹累乘出來的 soft 機率，不是單一 cut）。
- **YOLO 標註圖 + 裁切圖**：確認有沒有裁對側。
- **Grad-CAM 疊圖**（3 張，對應 topology 的 3 個 cut）：看模型在關注哪個區域，可用來檢查是否聚焦在股骨頭。

### 5.6 常見狀況
| 現象 | 原因 / 解法 |
| --- | --- |
| `未偵測到髖關節` | YOLO 沒抓到框。換清晰一點的圖，或確認是髖部 X 光 |
| 裁到錯的一側 | 左/右選反了，改選另一側重試 |
| `Models not initialized` | `_runtime.json` 不存在 → 先跑 `run_all.bat` |
| 改了模型但網頁沒變 | Flask 要重啟才會重讀 `_runtime.json` |

> 注意：若有訓練好的 combiner，demo 網頁的**最終 Stage 會改用學習型 combiner**（取代硬路由）。
> `07_update_web.py` 會在 `combiner[all]` 與 `combiner[topology3]` 之間**挑 OOF 較高者**寫入
> `_runtime.json`（通常是 `all`，用全部 10 個 cut），並自動把所需的 cut 全部載入網頁。
> 啟動時會印 `[boot] combiner = all / <model>`，predict 回傳的 `method` 會是 `combiner[all] <model>`。
> 沒有 combiner 時自動退回 hierarchy 硬路由。Grad-CAM 仍只畫 topology 的 3 個 cut。
> 候選分類器：logreg / SVM(linear,rbf) / RandomForest / HistGradientBoosting / XGBoost / LightGBM
> （XGBoost、LightGBM 需另外安裝；未安裝則自動略過）。

---

## 6. 一句話總結

`run_all.bat`（預設）= 用現有模型重算所有「誠實版」報表；要改善訓練品質得 `run_all.bat train` 重訓；
demo 網頁 = `run_all.bat web` 後開 `http://127.0.0.1:5000` 上傳 X 光看 Stage 與 Grad-CAM。
