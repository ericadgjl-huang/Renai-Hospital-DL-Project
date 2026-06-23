# 仁愛醫院 Ficat 4 階段 X 光分類

本專案把髖部 X 光影像分成 Ficat stage 1 到 stage 4。流程分成兩層：

1. 先把 4-class 問題拆成 10 個 binary cuts，例如 `12_vs_34`、`3_vs_4`。
2. 每個 cut 用 5-fold CV 訓練多個 backbone，再用 OOF 表現選出該 cut 的集成方式，最後搜尋 5 種 hierarchy topology。

目前新版流程是 **OOF-selected fold ensemble**：不再使用 `final/` 重新訓練模型做測試，而是讓選拔和測試都使用同一批 CV fold checkpoints。

---

## 2026-06 方法學更新（重要）

這批改動的目的是讓結果**誠實、可在論文中辯護**，並修掉幾個會虛報分數的問題：

1. **topology 不再用 test set 選**（修掉資料洩漏）。
   `06_search_hierarchy.py` 改成用 train_val 的 **OOF 4-class routing macro-F1** 選最佳 topology，
   outer test 只用來「回報」最終分數。`best_topology.json` 會同時記 `oof_macro_f1` 與 `test_macro_f1`，
   並標 `selected_by: oof_macro_f1`。

2. **bootstrap 95% 信賴區間**（`08_report_ci.py`）。
   test set 只有 ~58 張，單一數字沒有意義。產生 `outputs/report/ci_report.md`，
   例如 `macro-F1 0.668 (95% CI 0.531–0.782)`。並附人類 Ficat 判讀一致性 κ≈0.39–0.46 作為基準。

3. **學習型 combiner**（`09_train_combiner.py`，對應「3 刀 → 機率 → 分類器」的想法）。
   把每個 cut 的 `P(class=1)` 當特徵，用 LogReg / SVM(linear,rbf) / RandomForest / HistGradientBoosting
   做 4-class，**用 OOF 選模型、test 只回報**，並與硬路由 hierarchy 並排比較
   （`outputs/combiner/comparison.csv`）。可取代「一刀切錯就回不來」的硬路由。

4. **訓練正則化**（`train.py` / `cv.py`）。
   class-weighted cross-entropy + AdamW weight decay + early stopping，
   增強改為 affine + 亮度/對比 jitter（**刻意不水平翻轉**，因為右髖已被翻成左髖方向）。
   這些只在**重新訓練**（`04_train_all_cuts.py`）後生效。新增 CLI 旗標：`--weight-decay`、`--patience`。

5. **GroupKFold 準備**（`data.py`）。
   目前檔名 `S<stage>_<side><n>.jpg` **沒有病人 ID**，無法做病人層級切分（有潛在 patient leakage 風險）。
   `make_outer_split` / `make_cv_folds` 已支援可選 `groups` 參數，`patient_groups()` 會在
   `roi_all.csv` 出現 `patient` 欄位時自動啟用 `StratifiedGroupKFold`。**請向醫院索取 filename→病人 對照表。**

> 一鍵跑（先 `conda activate unet_labeling`）：`run_all.bat`（從既有 checkpoints 重建並產生上述全部報表）。

---

## 從 Drive 原始資料到最終輸出

以下指令假設你在專案根目錄執行，也就是有：

```text
drive-download-20251023T113302Z-1-001/
  Ficat stage 1/
  Ficat stage 2/
  Ficat stage 3/
  Ficat stage 4/
weights/yolo_best.pt
```

### 0. 建立環境

```powershell
conda env create -f environment.yaml
conda activate unet_labeling
pip install -e .
```

如果環境已經建好，只需要：

```powershell
conda activate unet_labeling
pip install -e .
```

### 1. 從 Drive 原圖產生分類資料集

這一步會用 YOLO 找 ROI，裁切後把右側影像翻成左側方向，輸出：

```text
stage_cls_dataset/stage_1..stage_4/
roi_all.csv
```

如果 `stage_cls_dataset/` 已經存在且你不想重做，可以跳過這步。

```powershell
python scripts/01_prepare_stage_dataset.py `
  --raw-root drive-download-20251023T113302Z-1-001 `
  --weights weights/yolo_best.pt `
  --out stage_cls_dataset `
  --roi-csv roi_all.csv `
  --device 0 `
  --overwrite
```

若沒有 GPU，把 `--device 0` 改成：

```powershell
--device cpu
```

### 2. 訓練全部 binary cuts

這一步很久，會訓練：

```text
10 cuts x 5 folds x 5 backbones = 250 個 fold checkpoints
```

```powershell
python scripts/04_train_all_cuts.py
```

輸出位置：

```text
outputs/cuts/<cut>/cv/fold_0/<backbone>/best_<backbone>.pth
outputs/cuts/<cut>/cv/fold_1/<backbone>/best_<backbone>.pth
...
outputs/cuts/<cut>/summary.csv
outputs/cuts/<cut>/cv_per_fold.csv
```

如果你只是測試流程，可以先跑 smoke test：

```powershell
python scripts/03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

### 3. 建立 OOF ensemble

如果你已經訓練過 CV checkpoints，這一步可以直接跑，不需要重跑 Step 2。

```powershell
python scripts/05_build_ensemble.py
```

每個 cut 會輸出：

```text
outputs/cuts/<cut>/ensemble/winner.json
outputs/cuts/<cut>/ensemble/test_probs.npy
outputs/cuts/<cut>/ensemble/confusion_matrix_test.png
outputs/cuts/<cut>/ensemble/classification_report_test.txt
```

總表：

```text
outputs/ensemble_summary.csv
```

如果 `val_probs.npy` 不存在也沒關係，`ensemble.py` 會載入 fold checkpoints 重新 inference validation fold，只是第一次會比較久。

### 4. 搜尋最佳 hierarchy topology

```powershell
python scripts/06_search_hierarchy.py
```

輸出：

```text
outputs/hierarchy/search_results.csv
outputs/hierarchy/best_topology.json
outputs/hierarchy/best_topology.yaml
outputs/hierarchy/T1..T5/
```

### 5. 同步 Web App runtime

```powershell
python scripts/07_update_web.py
```

這會產生：

```text
web_app/_runtime.json
```

### 6. 啟動 Web App

```powershell
python web_app/app.py
```

預設網址：

```text
http://127.0.0.1:5000
```

如果你已經開著 Flask，跑完 `07_update_web.py` 後要重啟 Flask，才會吃到新的 `_runtime.json`。

---

## 如果只想從既有 checkpoints 重建最終結果

當 `outputs/cuts/<cut>/cv/fold_*/<backbone>/best_<backbone>.pth` 已經存在時，不需要再跑 `04_train_all_cuts.py`。

可以先清掉舊 ensemble/hierarchy 結果：

```cmd
for /D %i in (outputs\cuts\*) do @if exist "%i\ensemble" rmdir /S /Q "%i\ensemble"
if exist outputs\hierarchy rmdir /S /Q outputs\hierarchy
if exist outputs\ensemble_summary.csv del /F /Q outputs\ensemble_summary.csv
```

然後直接跑：

```powershell
python scripts/05_build_ensemble.py
python scripts/06_search_hierarchy.py
python scripts/07_update_web.py
```

---

## 模型選拔與集成流程

### 1. 為什麼要拆成 binary cuts

Ficat stage 有 4 類。這個專案不是直接訓練一個 4-class classifier，而是把問題拆成多個二元問題：

| Cut | label 0 | label 1 |
| --- | --- | --- |
| `1_vs_234` | stage 1 | stage 2,3,4 |
| `12_vs_34` | stage 1,2 | stage 3,4 |
| `123_vs_4` | stage 1,2,3 | stage 4 |
| `2_vs_34` | stage 2 | stage 3,4 |
| `23_vs_4` | stage 2,3 | stage 4 |
| `1_vs_23` | stage 1 | stage 2,3 |
| `12_vs_3` | stage 1,2 | stage 3 |
| `2_vs_3` | stage 2 | stage 3 |
| `1_vs_2` | stage 1 | stage 2 |
| `3_vs_4` | stage 3 | stage 4 |

每個 hierarchy topology 會用其中 3 個 cuts 把樣本一路分到 stage 1、2、3、4。

### 2. 每個 cut 怎麼訓練

對每個 cut，程式會在 train_val 資料上做 5-fold CV。

每個 fold 會訓練 5 個 backbone：

```text
efficientnet_b0
efficientnet_b1
resnet50
convnext_tiny
convnext_small
```

所以每個 cut 會得到：

```text
5 folds x 5 backbones = 25 個 checkpoints
```

但實際**集成只取每個 fold 的 validation 最佳 backbone**，所以 ensemble 階段只會用：

```text
5 folds x 1 best backbone = 5 個 checkpoints
```

例如 `12_vs_34` 訓練後可能是：

```text
outputs/cuts/12_vs_34/cv/fold_0/<5 個 backbone>/best_*.pth
outputs/cuts/12_vs_34/cv/fold_1/<5 個 backbone>/best_*.pth
...
```

`ensemble.py` 會讀每個 fold 的 `val_metrics.json`，找出該 fold macro-F1 最高的 backbone，最終 5 個 fold 各拿 1 個，例如：

```text
fold 0: best=convnext_small  val_macro_f1=0.85
fold 1: best=efficientnet_b1 val_macro_f1=0.83
fold 2: best=convnext_small  val_macro_f1=0.86
fold 3: best=resnet50        val_macro_f1=0.81
fold 4: best=convnext_tiny   val_macro_f1=0.84
```

### 3. OOF 是什麼

OOF 是 out-of-fold prediction。

假設某張 train_val 影像在 fold 2 的 validation set 裡，那它的 OOF 預測只能由 fold 2 的模型產生。因為 fold 2 的模型訓練時沒有看過這張影像，所以這個預測比較接近「沒看過資料」時的表現。

這是新版流程的核心：**用 OOF 表現做選拔，不用 test set 做選拔。**

### 4. 每個 cut 會比較兩種集成方式

`scripts/05_build_ensemble.py` 會對每個 cut 比較：

兩種策略都只用 5 個 fold-winner ckpts，不會用到其他 20 個。

#### fold_voting

OOF 階段：

```text
每個樣本 s ∈ V_k 的 OOF prediction = fold k 的 best backbone 的 softmax(s)
（這個 model 訓練時沒看過 s，所以是 unbiased）
```

Test 階段：

```text
5 個 fold-winner 的 softmax 平均，再 argmax
```

#### stacking

OOF 階段：

```text
X_oof = 每個樣本 ∈ V_k 由 fold k 的 best backbone softmax 構成
shape = (N_trainval, 2)
```

再用：

```text
LogisticRegression(class_weight="balanced")
```

做 meta classifier。OOF 分數不是直接 fit 後評估，而是用：

```text
cross_val_predict(LogReg, X_oof, y_oof, cv=5)
```

這樣 voting 和 stacking 都是在同一批 OOF 預測上公平比較。

> 註：因為只有 5 個 model + 2-dim softmax，stacking 本質上等同 LR 校正過的閾值調整；多數情況下不會比 voting 強，但兩個 OOF 分數都會記錄在 `winner.json` 供事後檢查。

Test 階段：

```text
5 個 fold-winner 的 softmax 平均 → 餵給已重新 fit (使用全 X_oof) 的 meta → predict_proba
```

### 5. winner 怎麼決定

每個 cut 都只看 OOF macro-F1：

```text
如果 stacking OOF macro-F1 > voting OOF macro-F1
    chosen = stacking
否則
    chosen = fold_voting
```

結果寫在：

```text
outputs/cuts/<cut>/ensemble/winner.json
```

重要的是：`test_macro_f1` 只是報告用，不參與選拔。

### 6. 為什麼新版不再用 final retrain model

舊問題是：

```text
OOF 選拔依據：5 個 CV fold models
最後 test 使用：final/ 重新用完整 train_val 訓練的另一個 model
```

這會造成「選拔用的模型」和「上場考 test 的模型」不是同一種模型定義。

新版已改成：

```text
OOF 選拔依據：5 個 fold-winner CV checkpoints
最後 test 使用：同一批 5 個 fold-winner ckpts 的 ensemble
```

所以 selection 和 final evaluation 一致。

### 7. hierarchy topology 怎麼選

4 個 stage 的有序 binary tree 共有 5 種：

| Topology | 結構 | 使用 cuts |
| --- | --- | --- |
| T1 | `((1,2),(3,4))` | `12_vs_34`, `1_vs_2`, `3_vs_4` |
| T2 | `(1,(2,(3,4)))` | `1_vs_234`, `2_vs_34`, `3_vs_4` |
| T3 | `(1,((2,3),4))` | `1_vs_234`, `23_vs_4`, `2_vs_3` |
| T4 | `((1,(2,3)),4)` | `123_vs_4`, `1_vs_23`, `2_vs_3` |
| T5 | `(((1,2),3),4)` | `123_vs_4`, `12_vs_3`, `1_vs_2` |

`scripts/06_search_hierarchy.py` 會對 5 種 topology 都跑 outer test set，計算 4-class macro-F1，選最高者寫入：

```text
outputs/hierarchy/best_topology.json
```

### 8. Web App 用的是什麼

`scripts/07_update_web.py` 會把最佳 topology 和各 cut 的 winner 寫到：

```text
web_app/_runtime.json
```

Web App 啟動時會載入：

```text
weights/yolo_best.pt
web_app/_runtime.json
outputs/cuts/<cut>/cv/fold_*/<backbone>/best_<backbone>.pth
```

上傳影像後流程是：

```text
原始 X 光
  -> YOLO 裁 ROI
  -> 若是右側則水平翻轉
  -> 依最佳 topology 跑 3 個 binary cuts
  -> 輸出 Stage 1/2/3/4 與 Grad-CAM
```

---

## 主要輸出檔

```text
outputs/ensemble_summary.csv
outputs/cuts/<cut>/ensemble/winner.json
outputs/hierarchy/search_results.csv
outputs/hierarchy/best_topology.json
web_app/_runtime.json
```

---

## 注意事項

- `outputs/`、`stage_cls_dataset/`、`*.pth` 通常不要 push 到 Git。
- `outputs/splits/outer_split.json` 會固定 outer 80/20 split，重跑時會沿用同一份 split。
- 如果要完全重做實驗，才需要刪掉 `outputs/` 後從 Step 1 或 Step 2 開始。
- 如果只是重建 ensemble/hierarchy，保留 `outputs/cuts/*/cv/`，從 `05_build_ensemble.py` 開始即可。
