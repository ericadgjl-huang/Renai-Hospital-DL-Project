# 仁愛醫院 — 髖關節 X 光 Ficat 4 階段分級

用 5-fold cross-validation + **fold-level soft voting** + 窮舉 5 種 binary
tree 拓撲的階層式分類管線。每個 cut 直接把 25 個 per-fold 最佳模型
（5 backbone × 5 fold）做 soft voting 當作集成預測，不再做 final 重訓、
也不需要在「單模 vs 集成」之間做判斷。

---

## 系統架構（v3 — fold-voting）

```
原始 X 光 ──► YOLOv8 偵測 ROI ──► 統一翻轉成左側 ──►
   ┌────────────────────────────────────────────────┐
   │  最佳 binary-tree 拓撲（從 5 種中搜尋而得）       │
   │   每個 node = 1 個 binary cut（10 種候選之一）    │
   │   每個 cut  = 5 fold × 5 backbone (25 ckpt) 的    │
   │              soft-voting 集成                     │
   └────────────────────────────────────────────────┘
                          ▼
                Stage 1 / 2 / 3 / 4
```

### 為什麼這樣做

| 設計 | 動機 |
| --- | --- |
| 5-fold CV bagging | 每個 fold 的 validation-best ckpt 都是「沒看過該 fold val」的模型；soft-vote 25 個降低 variance，等同 cross-validation ensemble |
| 不再 final 重訓 | 取消 retrain on full 80%，因為 fold 模型本身就是 ensemble member；省下 50 次重訓時間 |
| 不再「單模 vs 集成」 | fold-voting 永遠是集成結果，沒有要不要集成的判斷 |
| 5 種拓撲窮舉 | 4 類別的 binary tree 共 5 種（Catalan(3)=5）；窮舉後用 4-class macro-F1 決定贏家 |
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
│   ├── models.py               # 5 個 backbone factory
│   ├── train.py                # train_one(...) — 一個 (backbone, fold) 的訓練
│   ├── cv.py                   # 一個 cut 的 5-fold CV（無 final 重訓）
│   ├── ensemble.py             # 5 fold × 5 backbone = 25 ckpt 的 soft voting
│   ├── hierarchy.py            # 5 拓撲 + 4-class 推論 + best topology 搜尋
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
│   │   ├── cv/fold_{0..4}/<backbone>/best_<backbone>.pth   # 25 ckpts/cut
│   │   ├── cv/fold_{0..4}/<backbone>/val_metrics.json
│   │   ├── summary.csv                                     # CV 指標
│   │   ├── cv_per_fold.csv
│   │   └── ensemble/
│   │       ├── winner.json                                 # 25 members 與 test 指標
│   │       ├── test_probs.npy
│   │       ├── confusion_matrix_test.png
│   │       └── classification_report_test.txt
│   └── hierarchy/
│       ├── search_results.csv
│       ├── best_topology.json
│       └── T1/, T2/, ...
├── stage_cls_dataset/          # ImageFolder（既有，stage_1..stage_4）
└── web_app/
    ├── app.py                  # Flask app（讀 _runtime.json）
    └── _runtime.json           # 由 07_update_web.py 自動產生
```

---

## 環境

```powershell
conda activate unet_labeling
# 或第一次安裝：
conda env create -f environment.yaml
```

可選：把套件 editable 安裝：

```powershell
pip install -e .
```

---

## 完整訓練流水線

> 全程在工作目錄根部下指令；固定 `seed=42`。

### Step 0 — 從 YOLO ROI 建 ImageFolder（歷史紀錄、目前可跳過）

`stage_cls_dataset/` 已經在 repo 中存在並被 v3 pipeline 直接使用。
`scripts/00_make_dataset.py` 仍保留原本的「YOLO crop + L/R 翻轉 + 寫成
ImageFolder」邏輯做為紀錄；它需要 git-ignored 的中間資料夾
`yolo_dataset_process/yolo_dataset/images/`，目前 checkout 不含此資料夾，
因此腳本會偵測到並印出 `[skip] ...` 後乾淨退出 — **預期行為**，不是錯誤。

```powershell
# 想再生 stage_cls_dataset 才需要跑（必須先有 yolo_dataset_process/）
python scripts/00_make_dataset.py
```

### Step 1 — 對單一 cut 跑 5-fold CV（先驗證流程）

```powershell
python scripts/03_train_cv.py --cut 1_vs_234
# Smoke test（1 fold × 2 epoch × 1 backbone）：
python scripts/03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

每個 (backbone, fold) 都會留下 validation-best ckpt：

```
outputs/cuts/<cut>/cv/fold_0/<backbone>/best_<backbone>.pth
                              val_metrics.json
                              confusion_matrix_val.png
                              classification_report_val.txt
outputs/cuts/<cut>/cv/fold_1/...
outputs/cuts/<cut>/summary.csv         # CV mean/std per backbone
outputs/cuts/<cut>/cv_per_fold.csv     # 逐 fold 詳細結果
```

### Step 2 — 跑完所有 10 個 cuts

```powershell
python scripts/04_train_all_cuts.py
# 只跑指定的：
python scripts/04_train_all_cuts.py --only 1_vs_234 2_vs_3
```

5 backbone × 5 fold × 10 cut = 250 次訓練；視 GPU 可能要數小時。

### Step 3 — 對每個 cut 做 25-model soft voting

```powershell
python scripts/05_build_ensemble.py
```

每個 cut 都會寫出 `outputs/cuts/<cut>/ensemble/winner.json`：

```json
{
  "decision": {
    "cut": "1_vs_234",
    "chosen": "fold_voting",
    "members": [
      {"backbone": "efficientnet_b0", "fold": 0, "ckpt": "outputs/cuts/.../best_efficientnet_b0.pth"},
      ...
    ],
    "test_macro_f1": 0.84,
    "test_accuracy": 0.85,
    "test_auc": 0.91
  },
  "n_members": 25
}
```

`outputs/ensemble_summary.csv` 會匯總每個 cut 的 test 指標。

### Step 4 — 5 種拓撲窮舉，挑最佳 4-class macro-F1

```powershell
python scripts/06_search_hierarchy.py
```

輸出：

```
outputs/hierarchy/
├── search_results.csv
├── best_topology.json
├── best_topology.yaml
└── T1/, T2/, ...
```

### Step 5 — 同步到 web_app

```powershell
python scripts/07_update_web.py
```

之後 `python web_app/app.py` 啟動時讀 `_runtime.json` 自動建路由；
每個 cut 都會在 cpu 上同時載入 ≤25 個 ckpt 做 soft voting。

---

## 重要設計細節

* **固定 seed=42 全程一致**。`src/renai/seed.py` 是唯一 seed 入口；
  CV 折之間用 `SEED+fold_index` 讓不同 fold 的 RNG 不同步但仍可重現。
* **outer split 一次寫死**：`outputs/splits/outer_split.json` 第一次跑時計算
  並快取。後續每個 cut、每次 ensemble、hierarchy 搜尋都讀同一份。
* **Cut-aware sample filtering**：`2_vs_3` / `1_vs_2` 訓練時只看自己範圍內
  的 stage（不看 stage 1/4）；hierarchy 推論時若上層 cut 把 stage 1 樣本
  誤送進 `2_vs_3`，那是階層錯誤，不是訓練錯誤。
* **Soft voting 永遠是集成結果**：不再有「單模 vs 集成」的選擇；
  fold-voting 直接出最終 test 指標。

---

## 快速 sanity check

```powershell
python scripts/03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

通過後再跑完整流程（Step 2 ~ 5）。

---

## Web 部署（保留 Render 流程）

`requirements-render.txt` / `runtime.txt` / `web_app/app.py` 仍可用 Render
部署。雲端版讀同一份 `_runtime.json`；25 ckpt 透過 `MODEL_ASSET_BASE_URL`
拉取（沿用原本的下載機制）。

---

## Git

```powershell
git status
git add <files>
git commit -m "<msg>"
git push
```

> ⚠️ 別把 `outputs/`、`stage_cls_dataset/`、`*.pth` push 上去 — 跟原本一樣
> 透過 GitHub Releases 或 R2/S3 分發權重。
