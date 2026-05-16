# 仁愛醫院 — 髖關節 X 光 Ficat 4 階段分級

利用 5-fold cross-validation、跨家族集成（EfficientNet / ResNet / ConvNeXt）
與「窮舉 5 種 binary tree 拓撲」的階層式分類管線，將髖關節影像自動分到
Ficat Stage 1 ~ 4。所有實驗都用固定 `seed=42`，每一個 cut / fold / backbone
都有獨立輸出資料夾以便逐一檢視結果。

---

## 系統架構（v2 — 重構版）

```
原始 X 光 ──► YOLOv8 偵測 ROI ──► 統一翻轉成左側 ──►
   ┌────────────────────────────────────────────────┐
   │  最佳 binary-tree 拓撲（從 5 種中搜尋而得）       │
   │   每個 node = 1 個 binary cut（10 種候選之一）    │
   │   每個 cut  = 跨家族 top-3 集成 OR 單一最強模型   │
   └────────────────────────────────────────────────┘
                          ▼
                Stage 1 / 2 / 3 / 4
```

### 為什麼這樣做

| 設計 | 動機 |
| --- | --- |
| 5-fold CV + 重訓 | 用 5 fold 平均找穩定的 best_epoch，再用整段 80% 重訓做最終模型，比單次切分更可靠 |
| top-3 跨家族集成 | EfficientNet / ResNet / ConvNeXt 抓到不同特徵；voting 與 stacking 兩種策略都試，唯有比單一最強模型更好才採用 |
| 5 種拓撲窮舉 | 4 類別的 binary tree 共 5 種（Catalan(3)=5）。之前固定 (123)/(4)→(1)/(23)→(2)/(3) 並未驗證過是否最優；窮舉後再以 4-class macro-F1 決定 |
| 固定 seed=42 | 切分、CV folds、初始化、cuDNN 都鎖住，所有實驗都可比較 |
| 個別 backbone 子資料夾 | `outputs/cuts/<cut>/cv/fold_X/<backbone>/` 每個訓練都自留 weights、log、混淆矩陣，實驗可追溯 |

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
| T3 | `(1,((2,3),4))`     | 1_vs_234, 23_vs_4, 2_vs_3（舊版本） |
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
│   ├── cv.py                   # 一個 cut 的 CV + 平均 best_epoch + final 重訓
│   ├── ensemble.py             # 跨家族 top-3 + voting / stacking + 勝者選擇
│   ├── hierarchy.py            # 5 拓撲 + 4-class 推論 + best topology 搜尋
│   ├── cuts_registry.py        # 10 cuts 註冊
│   ├── eval.py                 # 指標 / CM / report / json
│   ├── gradcam.py              # Grad-CAM utility
│   └── cli/                    # CLI entry points
├── scripts/                    # 一鍵腳本（不必 pip install）
│   ├── 00_make_dataset.py
│   ├── 03_train_cv.py
│   ├── 04_train_all_cuts.py
│   ├── 05_build_ensemble.py
│   ├── 06_search_hierarchy.py
│   └── 07_update_web.py
├── outputs/                    # 全部訓練輸出（per-cut / per-fold / per-backbone）
├── stage_cls_dataset/          # ImageFolder（既有，stage_1..stage_4）
└── web_app/
    ├── app.py                  # Flask app（讀 _runtime.json 自動配置）
    └── _runtime.json           # 由 07_update_web.py 自動產生
```

舊的 `03_*.ipynb`、`outputs_bin*/`、`03.5_combination/` 已不會被新 pipeline
讀寫，可以保留作為對照或備份。

---

## 環境

跟原本一樣使用 Anaconda `unet_labeling` env：

```powershell
conda activate unet_labeling
# 或第一次安裝：
conda env create -f environment.yaml
```

可選：把套件 editable 安裝（之後就能直接 `python -m renai.cli.train_cv ...`）

```powershell
pip install -e .
```

---

## 完整訓練流水線

> ⚠️ 建議用 `unet_labeling` env 的 python（`C:/Users/USER/anaconda3/envs/unet_labeling/python.exe`），並全程在工作目錄根部下指令。所有指令都用固定 `seed=42`，可重現。

### Step 0 — 從 YOLO ROI 建 ImageFolder（已存在則可省略）

```powershell
python scripts/00_make_dataset.py
# 產生 stage_cls_dataset/stage_1..stage_4/
```

### Step 1 — 對單一 cut 跑 5-fold CV + final 重訓（先驗證流程）

```powershell
python scripts/03_train_cv.py --cut 1_vs_234
# 想快速驗證 wiring：
python scripts/03_train_cv.py --cut 3_vs_4 --smoke
```

輸出在 `outputs/cuts/<cut>/`：

```
cv/fold_0/<backbone>/   best.pth, train_log.csv, val_metrics.json, CM, report
cv/fold_1/<backbone>/   …
…
final/<backbone>/       重訓 ckpt + test_metrics.json + CM + report + test_probs.npy
summary.csv             每個 backbone 的 cv mean/std + test 指標
cv_per_fold.csv         逐 fold 結果，方便畫表
```

### Step 2 — 跑完所有 10 個 cuts（這步耗時）

```powershell
python scripts/04_train_all_cuts.py
# 只跑指定的：
python scripts/04_train_all_cuts.py --only 1_vs_234 2_vs_3
```

5 backbone × 5 fold × 10 cut = 250 次訓練 + 50 次重訓；視 GPU 可能要數小時。
失敗的 cut 不會擋下其他 cut（`--only` 之後可單獨補跑）。

### Step 3 — 對每個 cut 建集成 vs 單一最強的決策

```powershell
python scripts/05_build_ensemble.py
```

輸出：每個 `outputs/cuts/<cut>/ensemble/winner.json`，記錄

```json
{
  "candidates": { "single": {...}, "voting": {...}, "stacking": {...} },
  "decision":   { "chosen": "stacking", "chosen_members": [...], "test_macro_f1": ... }
}
```

> 規則：voting / stacking 唯有 `test_macro_f1` 嚴格大於單一最強才會勝出；否則回退到單一模型。對應你的需求「沒變好就用單一模型」。

> 重要：stacking 的 meta classifier 是用 OOF（out-of-fold）特徵訓練，不是用 final 模型在 val 上重新跑（避免資料洩漏）。test 階段才用 final 模型。

### Step 4 — 5 種拓撲窮舉，挑最佳 4-class macro-F1

```powershell
python scripts/06_search_hierarchy.py
```

輸出：

```
outputs/hierarchy/
├── search_results.csv           # 5 拓撲全部指標
├── best_topology.json           # 勝出拓撲 + 對應 cuts
├── best_topology.yaml           # 同上 YAML 版（給人看）
└── T1/, T2/, …                  # 每個拓撲的 4-class CM + report + metrics
```

### Step 5 — 同步到 web_app

```powershell
python scripts/07_update_web.py
# 寫出 web_app/_runtime.json
```

之後 `python web_app/app.py` 啟動時就會自動讀 `_runtime.json`，根據其中
`topology` 與 `cuts` 構建路由器。完全不必再改 `app.py`。

---

## 重要設計細節

* **固定 seed=42 全程一致**。`src/renai/seed.py` 是唯一 seed 入口；
  CV 折之間用 `SEED+fold_index` 讓不同 fold 的 RNG 不同步但仍可重現。
* **outer split 一次寫死**：`outputs/splits/outer_split.json` 第一次跑時計算
  並快取。後續每個 cut、每次 ensemble、hierarchy 搜尋都讀同一份。重跑要換
  split 的話刪掉這個檔。
* **Cut-aware sample filtering**：`2_vs_3`、`1_vs_2` 這種 cut 訓練時只看自己
  範圍內的 stage（不看 stage 1/4），避免無關資料汙染。階層推論時若上層
  cut 把 stage 1 樣本誤判給 `2_vs_3`，那是階層錯誤，不是訓練錯誤。
* **集成優於單一才採用**：嚴格比較 `test_macro_f1`，平手或變差就用單一模型。

---

## 快速 sanity check

```powershell
python scripts/03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
# 1 fold × 2 epochs × 1 backbone，幾十秒驗 wiring
```

通過後再跑完整流程（Step 2 ~ 5）。

---

## Web 部署（保留 Render 流程）

`requirements-render.txt` / `runtime.txt` / `web_app/app.py` 仍可用 Render 部
署。雲端版讀同一份 `web_app/_runtime.json`，只是模型權重要透過
`MODEL_ASSET_BASE_URL` 拉取（沿用原本的下載機制）。

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
