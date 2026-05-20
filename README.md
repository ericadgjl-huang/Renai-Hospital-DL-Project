# Renai Hospital Ficat Stage Classifier

這個專案把髖部 X 光分成 Ficat stage 1 到 stage 4。主要流程是：

```text
drive-download-20251023T113302Z-1-001/
  Ficat stage 1..4 原始 X 光
        |
        |  YOLO 偵測 ROI / 產生 roi_all.csv
        v
stage_cls_dataset/
  stage_1..stage_4 分類資料夾
        |
        |  10 個 binary cuts 的 5-fold CV
        v
outputs/cuts/<cut>/
        |
        |  OOF 選 single / voting / stacking
        v
outputs/cuts/<cut>/ensemble/winner.json
        |
        |  5 種 hierarchy topology 用 OOF 選勝者
        v
outputs/hierarchy/best_topology.json
```

目前分類訓練真正讀取的是 `stage_cls_dataset/`。`drive-download-20251023T113302Z-1-001/` 是最原始影像來源；若 `stage_cls_dataset/` 已經存在，可以直接從「一鍵跑完整分類流程」開始。

## 環境安裝

建議在 Anaconda Prompt 或 PowerShell 中執行：

```cmd
conda env create -f environment.yaml
conda activate unet_labeling
pip install -e .
```

如果環境已經建好，只需要：

```cmd
conda activate unet_labeling
pip install -e .
```

## 從 Drive 原始資料到最終輸出

### 0. 確認原始資料位置

專案根目錄下應該有：

```text
drive-download-20251023T113302Z-1-001/
  Ficat stage 1/
  Ficat stage 2/
  Ficat stage 3/
  Ficat stage 4/
```

### 1. 產生或確認 `stage_cls_dataset/`

如果 `stage_cls_dataset/stage_1..stage_4` 已經存在且裡面有影像，這一步可以跳過。

```cmd
python scripts\00_make_dataset.py
```

注意：`scripts\00_make_dataset.py` 需要 `roi_all.csv` 與 YOLO 中繼影像資料夾：

```text
yolo_dataset_process/yolo_dataset/images/
roi_all.csv
```

這兩者是由歷史 YOLO 流程產生的。若要從 `drive-download-20251023T113302Z-1-001` 完整重建 ROI，請先依序跑：

```text
notebooks/01_yolo_train.ipynb
notebooks/02_yolo_infer_make_ROI.ipynb
```

完成後再執行：

```cmd
python scripts\00_make_dataset.py ^
  --roi-csv roi_all.csv ^
  --yolo-images yolo_dataset_process\yolo_dataset\images ^
  --out stage_cls_dataset
```

### 2. 一鍵跑完整分類流程

這會訓練 10 個 binary cuts，每個 cut 跑 5 個 backbone、5-fold CV，接著建立 ensemble，最後搜尋最佳 hierarchy。

```cmd
python scripts\04_train_all_cuts.py --data-root stage_cls_dataset --out-root outputs --epochs 30 --batch-size 16
python scripts\05_build_ensemble.py --data-root stage_cls_dataset --out-root outputs --batch-size 16
python scripts\06_search_hierarchy.py --data-root stage_cls_dataset --out-root outputs --batch-size 16
python scripts\07_update_web.py --out-root outputs
```

最終主要輸出在：

```text
outputs/hierarchy/search_results.csv
outputs/hierarchy/best_topology.json
outputs/hierarchy/best_topology.yaml
outputs/hierarchy/<winner>/classification_report_test.txt
outputs/hierarchy/<winner>/confusion_matrix_test.png
web_app/_runtime.json
```

### 3. 只跑小測試

如果只是確認程式能不能跑，不想訓練完整模型：

```cmd
python scripts\03_train_cv.py --cut 3_vs_4 --backbones efficientnet_b0 --smoke
```

### 4. 只重跑部分 cut

```cmd
python scripts\04_train_all_cuts.py --only 1_vs_234 2_vs_3 --epochs 30
python scripts\05_build_ensemble.py --cut 1_vs_234
python scripts\05_build_ensemble.py --cut 2_vs_3
python scripts\06_search_hierarchy.py
```

## 現在的模型選拔與集成流程

### 1. 外圈資料切分

程式先用 4-class label 做固定的 80/20 stratified split：

```text
80% train_val：訓練、CV、OOF 選擇
20% test：最後只給勝出的 hierarchy 做一次最終評估
```

切分結果存在：

```text
outputs/splits/outer_split.json
```

### 2. 10 個 binary cuts

原本是 4 類分類問題，但 hierarchy 會拆成多個二分類節點。程式定義了 10 個 binary cuts：

| Cut | label 0 | label 1 |
| --- | --- | --- |
| `1_vs_234` | stage 1 | stage 2/3/4 |
| `12_vs_34` | stage 1/2 | stage 3/4 |
| `123_vs_4` | stage 1/2/3 | stage 4 |
| `2_vs_34` | stage 2 | stage 3/4 |
| `23_vs_4` | stage 2/3 | stage 4 |
| `1_vs_23` | stage 1 | stage 2/3 |
| `12_vs_3` | stage 1/2 | stage 3 |
| `2_vs_3` | stage 2 | stage 3 |
| `1_vs_2` | stage 1 | stage 2 |
| `3_vs_4` | stage 3 | stage 4 |

例如 `2_vs_3` 只會拿 stage 2 和 stage 3 的樣本訓練，不會把 stage 1 或 stage 4 混進去。

### 3. 每個 cut 的 5-fold CV

每個 cut 會訓練 5 個 backbone：

```text
efficientnet_b0
efficientnet_b1
resnet50
convnext_tiny
convnext_small
```

每個 backbone 會跑 5-fold CV。每個 fold 會用 validation macro-F1 挑 best checkpoint：

```text
outputs/cuts/<cut>/cv/fold_0/<backbone>/best_<backbone>.pth
outputs/cuts/<cut>/cv/fold_1/<backbone>/best_<backbone>.pth
...
```

同時會用 5 個 fold 的 `best_epoch` 平均值，在完整 80% train_val 上重新訓練一個 final model：

```text
outputs/cuts/<cut>/final/<backbone>/best_<backbone>.pth
```

### 4. OOF 預測

OOF 是 out-of-fold。意思是每個 train_val 樣本都用「訓練時沒看過它」的 fold model 做預測。

對某個 cut 和 backbone 來說，OOF 預測會組成：

```text
outputs/cuts/<cut>/oof/all_train_val.npz
```

這個檔案會被拿來比較 backbone、ensemble 和 hierarchy topology。這樣做的重點是：選模型時不看 20% test。

### 5. 每個 cut 選 single / voting / stacking

對每個 cut，程式會比較三種策略：

| 策略 | 說明 |
| --- | --- |
| `single` | 選 OOF macro-F1 最高的單一 backbone |
| `voting` | 從 EfficientNet / ResNet / ConvNeXt 家族各挑高分模型，取 top-3 soft voting |
| `stacking` | 用 top-3 的 OOF probability 當特徵，訓練 LogisticRegression meta model |

選擇規則是：

```text
如果 voting 或 stacking 的 OOF macro-F1 嚴格大於 single，才選 ensemble。
否則保留 single。
```

結果存在：

```text
outputs/cuts/<cut>/ensemble/winner.json
outputs/cuts/<cut>/oof/oof_p_class1.npy
```

`oof_p_class1.npy` 是這個 cut 勝出策略對每個 train_val 樣本輸出的 `P(label 1)`，後面 hierarchy 會用它來走樹。

### 6. 5 種 hierarchy topology 窮舉

4 個 stage 的有序 binary tree 一共有 5 種：

| Topology | 結構 | 使用 cuts |
| --- | --- | --- |
| T1 | `((1,2),(3,4))` | `12_vs_34`, `1_vs_2`, `3_vs_4` |
| T2 | `(1,(2,(3,4)))` | `1_vs_234`, `2_vs_34`, `3_vs_4` |
| T3 | `(1,((2,3),4))` | `1_vs_234`, `23_vs_4`, `2_vs_3` |
| T4 | `((1,(2,3)),4)` | `123_vs_4`, `1_vs_23`, `2_vs_3` |
| T5 | `(((1,2),3),4)` | `123_vs_4`, `12_vs_3`, `1_vs_2` |

每個 topology 會用各 cut 的 OOF `P(label 1)` 對 train_val 樣本走完整棵樹，得到 4-class OOF prediction。最後用 4-class OOF macro-F1 選 topology。

### 7. 最終 test 評估

選出最佳 topology 後，程式才會對 20% test 做一次最終評估，輸出：

```text
outputs/hierarchy/best_topology.json
outputs/hierarchy/<winner>/metrics_test.json
outputs/hierarchy/<winner>/classification_report_test.txt
outputs/hierarchy/<winner>/confusion_matrix_test.png
```

目前程式的 test inference 會載入每個 cut 的 `final/<backbone>/best_<backbone>.pth`。也就是說，OOF 選擇是根據 CV fold models，但最終 test 是用 full train_val retrain 後的 final models。這點是現在準確率變差時最需要留意的地方，因為 OOF 排名和 final model 的 test 排名可能不一致。

## 輸出檔案速查

```text
outputs/cuts/<cut>/summary.csv
  每個 backbone 的 CV mean/std、final test 指標

outputs/cuts/<cut>/cv_per_fold.csv
  每個 fold 的 validation 指標

outputs/cuts/<cut>/ensemble/winner.json
  該 cut 選 single / voting / stacking 的結果

outputs/hierarchy/search_results.csv
  5 種 topology 的 OOF macro-F1

outputs/hierarchy/best_topology.json
  最佳 topology 與最終 test 指標
```

## 啟動 Web App

先同步最新 hierarchy 設定：

```cmd
python scripts\07_update_web.py --out-root outputs
```

再啟動 Flask：

```cmd
python web_app\app.py
```

## Git 注意事項

通常不要把這些大型或產出檔 commit：

```text
outputs/
stage_cls_dataset/
weights/
*.pth
*.pkl
```
