# 仁愛醫院 — 髖關節 X 光 Ficat 4 階段分級

5-fold cross-validation + **OOF-based 集成選擇**（single / voting / stacking
三候選比較）+ 窮舉 5 種 binary tree 拓撲的階層式分類管線。所有模型 /
策略 / 拓撲的選擇都**只看 OOF（out-of-fold）validation 結果**，20% 外圈
test 集**只在最後勝者拓撲評估時用一次**，得到無偏的最終數字。

---

## 系統架構

```
原始 X 光 ──► YOLOv8 偵測 ROI ──► 統一翻轉成左側 ──►
   ┌────────────────────────────────────────────────┐
   │  最佳 binary-tree 拓撲（5 種拓撲用 OOF 挑出）     │
   │   每個 node = 1 個 binary cut（10 種候選之一）    │
   │   每個 cut  = single / voting / stacking 三候選   │
   │              用 5-fold OOF macro-F1 挑出最佳       │
   │              （沒贏過 single 就用 single）          │
   └────────────────────────────────────────────────┘
                          ▼
                Stage 1 / 2 / 3 / 4
```

### 為什麼這樣做

| 設計 | 動機 |
| --- | --- |
| 5-fold CV | 用 fold-level validation 找穩定的 best_epoch，並透過 OOF 預測無偏地評估模型 |
| 跨家族 top-3 集成 | EfficientNet / ResNet / ConvNeXt 抓到不同特徵；voting 與 stacking 兩種策略都試 |
| OOF 選擇 | 「選哪個策略 / 哪個拓撲」全部用 OOF macro-F1 決定，test 集不參與選擇 |
| Stacking 用 5-fold CV-of-OOF 評估 | meta 不會看到自己的訓練樣本，stacking 的 OOF 分數也無偏 |
| Final 重訓 | 用 mean(best_epoch) 把整段 80% 全部訓完，當作部署用權重；test 階段才用這個推論 |
| 集成優於 single 才採用 | voting / stacking 嚴格 > single 才換掉，否則用 single |
| 5 種拓撲窮舉 | 4 類別的 binary tree 共 5 種（Catalan(3)=5）；用 4-class OOF macro-F1 決定贏家 |
| 固定 seed=42 | 切分、CV folds、初始化、cuDNN 都鎖住 |

### 10 個 binary cuts（5 拓撲共用）

| Cut | left (label 0) | right (label 1) | 哪個拓撲用到 |
| --- | --- | --- | --- |
| `1_vs_234`  | {1}     | {2,3,4} | T2, T3 |
| `12_vs_34`  | {1,2}   | {3,4}   | T1 |
| `123_vs_4`  | {1,2,3} | {4}     | T4, T5 |
| `2_vs_34`   | {2}     | {3,4}   | T2 |
| `23_vs_4`   | {2,3}   | {4}     | T3 |
| `1_vs_23`   | {1}     | {2,3}   | T4 |
| `12_vs_3`   | {1,2}   | {3}     | T5 |
| `2_vs_3`    | {2}     | {3}     | T3, T4 |
| `1_vs_2`    | {1}     | {2}     | T1, T5 |
| `3_vs_4`    | {3}     | {4}     | T1, T2 |

### 5 種拓撲

| 拓撲 | 結構 | 需要的 cuts |
| --- | --- | --- |
| T1 | `((1,2),(3,4))`     | 12_vs_34, 1_vs_2, 3_vs_4 |
| T2 | `(1,(2,(3,4)))`     | 1_vs_234, 2_vs_34, 3_vs_4 |
| T3 | `(1,((2,3),4))`     | 1_vs_234, 23_vs_4, 2_vs_3 |
| T4 | `((1,(2,3)),4)`     | 123_vs_4, 1_vs_23, 2_vs_3 |
| T5 | `(((1,2),3),4)`     | 123_vs_4, 12_vs_3, 1_vs_2 |

---

## 專案目錄

```
.
├── configs/                    # YAML configs（base, cuts/*, topologies）
├── src/renai/                  # 套件
│   ├── seed.py                 # set_seed(42) 唯一入口
│   ├── data.py                 # 80/20 split + 5-fold + cut-aware filter
│   ├── models.py               # 5 個 backbone factory + family map
│   ├── train.py                # train_one(...) — 一個 (backbone, fold) 的訓練
│   ├── cv.py                   # CV + final 重訓 + per-backbone test eval
│   ├── ensemble.py             # OOF-based single/voting/stacking + 勝者決策
│   ├── hierarchy.py            # 5 拓撲 OOF 比較 + 勝者拓撲做一次 test
│   ├── cuts_registry.py        # 10 cuts 註冊
│   ├── eval.py                 # 指標 / CM / report / json
│   ├── gradcam.py              # Grad-CAM utility
│   └── cli/                    # CLI entry points
├── scripts/                    # 一鍵腳本
│   ├── 00_make_dataset.py
│   ├── 03_train_cv.py
│   ├── 04_train_all_cuts.py
│   ├── 05_build_ensemble.py
│   ├── 06_search_hierarchy.py
│   └── 07_update_web.py
├── outputs/
│   ├── splits/outer_split.json
│   ├── cuts/<cut>/
│   │   ├── cv/fold_{0..4}/<backbone>/best_<backbone>.pth
│   │   ├── cv/fold_{0..4}/<backbone>/val_metrics.json + CM + report
│   │   ├── final/<backbone>/best_<backbone>.pth        # 重訓於整段 80%
│   │   ├── final/<backbone>/test_metrics.json + CM + report
│   │   ├── summary.csv                                 # CV 指標 + 診斷用 test
│   │   ├── cv_per_fold.csv
│   │   ├── oof/all_train_val.npz                       # 每個 backbone 的 OOF
│   │   ├── oof/oof_p_class1.npy                        # 勝者策略的 OOF P(class=1)
│   │   └── ensemble/
│   │       ├── winner.json                             # 決策 + candidates（無 test）
│   │       ├── meta_logreg.pkl                         # stacking 部署用 meta
│   │       ├── confusion_matrix_oof.png
│   │       └── classification_report_oof.txt
│   └── hierarchy/
│       ├── search_results.csv                          # 5 拓撲 OOF 對照
│       ├── best_topology.json                          # 勝者 + 唯一的 test 數字
│       ├── best_topology.yaml
│       └── T1/, T2/, ...                               # 每拓撲 OOF CM；勝者另有 test CM
├── stage_cls_dataset/          # ImageFolder（gitignored）
├── weights/yolo_best.pt        # YOLO 偵測模型（gitignored）
└── web_app/
    ├── app.py                  # Flask app（讀 _runtime.json 自動配置）
    └── _runtime.json           # 由 07_update_web.py 自動產生
```

---

## 環境

```cmd
conda activate unet_labeling
```

第一次安裝：

```cmd
conda env create -f environment.yaml
```

可選：editable 安裝套件

```cmd
pip install -e .
```

---

## 完整訓練流水線

全程 Anaconda Prompt，固定 `seed=42`。

### Step 0 — 從 YOLO ROI 建 ImageFolder（歷史紀錄，可跳過）

`stage_cls_dataset/` 通常已存在。`scripts/00_make_dataset.py` 留作歷史紀錄；
若 `yolo_dataset_process/` 不存在會友善地印出 `[skip]` 後退出。

```cmd
python scripts\00_make_dataset.py
```

### Step 1 — 對單一 cut 跑 CV + final 重訓（先驗證流程）

```cmd
python scripts\03_train_cv.py --cut 1_vs_234
:: Smoke test：
python scripts\03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

每個 cut 產出：

```
outputs/cuts/<cut>/cv/fold_0/<backbone>/best_<backbone>.pth  + val_metrics + CM + report
outputs/cuts/<cut>/cv/fold_1/...
outputs/cuts/<cut>/final/<backbone>/best_<backbone>.pth      + test_metrics + CM + report
outputs/cuts/<cut>/summary.csv         # 每個 backbone 的 CV mean/std + 診斷 test
outputs/cuts/<cut>/cv_per_fold.csv     # 逐 fold 詳細
```

> ⚠️ `summary.csv` 裡的 test 欄位**僅供診斷**，**不會**用來選 ensemble 策略。
> 選擇全部走 OOF（見 Step 3）。

### Step 2 — 跑完所有 10 個 cuts

```cmd
python scripts\04_train_all_cuts.py
:: 補跑指定 cut：
python scripts\04_train_all_cuts.py --only 1_vs_234 2_vs_3
```

5 backbone × 5 fold × 10 cut + 5 backbone × 10 cut 重訓 = ~300 次訓練；
過夜或半天。

### Step 3 — 對每個 cut 用 OOF 決定 single / voting / stacking

```cmd
python scripts\05_build_ensemble.py
```

對每個 cut：
1. 用 5 個 fold ckpts 算每個 backbone 的 OOF 預測（in-cut 樣本用「沒看過自己」
   的那個 fold model；out-of-cut 樣本用 5 fold 平均）。
2. `single`：OOF macro-F1 最高的 backbone。
3. `voting`：跨家族 top-3 OOF 軟投票 macro-F1。
4. `stacking`：跨家族 top-3 的 OOF 特徵接 LogisticRegression；用 5-fold CV-of-OOF
   無偏評估。
5. **決策**：max(voting, stacking) 嚴格 > single 才採用，否則 single。

`outputs/cuts/<cut>/ensemble/winner.json`：

```json
{
  "cut": "1_vs_234",
  "selection_metric": "5-fold OOF macro-F1 (no test set used)",
  "candidates": {
    "single":   {"backbone": "convnext_tiny", "macro_f1": 0.83, ...},
    "voting":   {"members": [...],            "macro_f1": 0.85, ...},
    "stacking": {"members": [...],            "macro_f1": 0.86, ...}
  },
  "decision": {"chosen": "stacking", "oof_macro_f1": 0.86, ...}
}
```

> **沒有 `test_*` 欄位** — test 集在這一步完全沒被碰。

### Step 4 — 5 種拓撲窮舉，**OOF 挑勝者 + 單次 test 報告**

```cmd
python scripts\06_search_hierarchy.py
```

1. 對每個拓撲，用每個 cut 的勝者 OOF P(class=1) 把訓練樣本路由過樹，算 4-class
   OOF macro-F1。
2. 挑出 OOF macro-F1 最高的拓撲。
3. **只**對勝者拓撲用 final 重訓 ckpt + stacking meta 在 20% test 集跑一次推論，
   產出唯一的無偏 test macro-F1。

`outputs/hierarchy/`：

```
search_results.csv         # 5 拓撲 OOF 對照
best_topology.json         # 勝者 + 唯一的 test 數字
best_topology.yaml
T1/confusion_matrix_oof.png ... T5/confusion_matrix_oof.png
<winner>/confusion_matrix_test.png   # 最終 4-class CM
<winner>/classification_report_test.txt
```

`best_topology.json` 範例：

```json
{
  "name": "T3",
  "description": "(1,((2,3),4))",
  "oof_macro_f1": 0.75,    // 選拓撲用這個
  "test_macro_f1": 0.79    // 對勝者跑 1 次 test 得到的最終要報的數字
}
```

### Step 5 — 同步到 web_app

```cmd
python scripts\07_update_web.py
```

寫 `web_app/_runtime.json`，內容包含勝者拓撲與各 cut 的 single/voting/stacking
決策。`python web_app/app.py` 啟動時讀此檔自動建路由（YOLO ROI → 階層推論）。

---

## 重要設計細節

* **OOF（out-of-fold）是核心概念**：5-fold CV 中，每個 train_val 樣本都有「沒看過
  自己的那個 fold 模型」的預測；把這 5 折預測串起來 = OOF 預測陣列。可以拿來
  做模型 / 策略 / 拓撲選擇而**完全不用 test 集**。
* **Test 集只用一次**：唯一的 test 推論發生在 Step 4 的勝者拓撲評估，那個數字
  才是無偏的最終 macro-F1。
* **固定 seed=42 全程一致**。`src/renai/seed.py` 是唯一 seed 入口；CV 折之間用
  `SEED+fold_index` 讓不同 fold 的 RNG 不同步但仍可重現。
* **outer split 一次寫死**：`outputs/splits/outer_split.json` 第一次跑時計算並
  快取。後續所有步驟讀同一份。
* **Cut-aware sample filtering**：`2_vs_3` / `1_vs_2` 訓練時只看自己範圍內的
  stage；階層推論時上層 cut 把錯誤 stage 誤送進子 cut，那是階層錯誤不是訓練
  錯誤。
* **Stacking 的兩層 OOF**：base 模型用 fold CV 算 OOF 特徵；meta 用 5-fold CV-of-OOF
  評估自己。兩層都不會洩漏。
* **集成優於 single 才採用**：嚴格 `>` 比較，平手或變差就回退到 single。

---

## 快速 sanity check

```cmd
python scripts\03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

跑得通就可進 Step 2。

---

## Web 部署（Render）

`requirements-render.txt` / `runtime.txt` / `web_app/app.py` 仍可用 Render
部署。雲端版讀同一份 `_runtime.json`，模型權重透過 `MODEL_ASSET_BASE_URL`
拉取（沿用原本下載機制）。

---

## Git

```cmd
git status
git add <files>
git commit -m "<msg>"
git push
```

> ⚠️ 別把 `outputs/`、`stage_cls_dataset/`、`weights/`、`*.pth`、`*.pkl` push 上去 —
> 用 GitHub Releases 或 R2/S3 分發權重。
