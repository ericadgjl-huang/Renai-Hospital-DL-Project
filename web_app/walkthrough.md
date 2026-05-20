# Web App Walkthrough

這份文件說明目前 `web_app/` 的用途、啟動方式，以及它如何使用訓練流程產生的最佳 hierarchy 模型。

## 目前版本重點

目前 web app 不再寫死舊版的 `m1 / m2 / m3` 模型組合。它會讀取：

```text
web_app/_runtime.json
```

這個檔案由下面指令產生：

```cmd
python scripts\07_update_web.py --out-root outputs
```

`_runtime.json` 會記錄：

- 最佳 hierarchy topology，例如 `T1 ((1,2),(3,4))`
- 勝出 topology 需要用到哪些 binary cuts
- 每個 cut 最後選到 `single`、`voting` 或 `stacking`
- 每個 cut 對應的 final model 權重位置
- 若該 cut 使用 stacking，會記錄 `meta_logreg.pkl` 位置

所以只要重新訓練、重新 build ensemble、重新 search hierarchy，再跑一次 `07_update_web.py`，web app 就會改用最新模型設定。

## 啟動前檢查

請確認以下檔案存在：

```text
outputs/hierarchy/best_topology.json
web_app/_runtime.json
weights/yolo_best.pt
```

如果 `web_app/_runtime.json` 不存在，先執行：

```cmd
python scripts\07_update_web.py --out-root outputs
```

如果 `weights/yolo_best.pt` 不存在，app 會退回使用：

```text
weights/yolov8n.pt
```

但正式使用建議放入專案訓練好的 `yolo_best.pt`。

## 啟動方式

在專案根目錄執行：

```cmd
conda activate unet_labeling
python web_app\app.py
```

接著開啟：

```text
http://127.0.0.1:5000
```

## 使用流程

1. 上傳 X 光影像。
2. 選擇要分析的側別：左側 `L` 或右側 `R`。
3. App 使用 YOLO 偵測 ROI。
4. 若選擇右側 `R`，ROI 會水平翻轉，讓分類模型統一看成左側方向。
5. App 將 ROI 丟入目前最佳 hierarchy。
6. 頁面會顯示：
   - YOLO 偵測框影像
   - 裁切後的 ROI
   - 最終預測 stage
   - stage 1 到 stage 4 的機率
   - 每個 hierarchy cut 的 Grad-CAM overlay

## 推論流程

推論時的流程如下：

```text
上傳影像
  |
  v
YOLO 偵測 ROI
  |
  v
依 L/R 統一方向
  |
  v
讀取 _runtime.json 指定的最佳 topology
  |
  v
依 topology 的 rules 逐個 cut 取得 P(label 1)
  |
  v
把每條路徑的條件機率相乘，得到 stage 1..4 機率
  |
  v
選機率最高者作為最終 stage
```

每個 cut 的預測方式會依 `winner.json` 決定：

| Winner | Web app 推論方式 |
| --- | --- |
| `single` | 載入單一 final backbone |
| `voting` | 載入多個 final backbones，平均 softmax probability |
| `stacking` | 載入多個 final backbones，將 probability 串接後交給 LogisticRegression meta model |

## L/R 方向處理

訓練資料在建立 `stage_cls_dataset/` 時，會把右側 ROI 翻成左側方向。Web app 推論時也維持同樣規則：

- 使用者選 `L`：直接使用裁切 ROI。
- 使用者選 `R`：先水平翻轉 ROI，再送進分類模型。

Grad-CAM 顯示時會再對右側 overlay 做相對應處理，讓畫面位置和原始 ROI 對得起來。

## Grad-CAM 說明

每個 cut 會產生一張 Grad-CAM。若該 cut 是 `voting` 或 `stacking`，目前 app 會使用該 cut 的第一個 member backbone 產生 Grad-CAM，作為可視化參考。

Grad-CAM 用來輔助觀察模型關注區域，不等於模型唯一判斷依據。

## 常見問題

### `/predict` 回傳 models not initialized

通常是 `web_app/_runtime.json` 不存在，或裡面指向的模型權重不存在。請先跑：

```cmd
python scripts\07_update_web.py --out-root outputs
```

並確認 `outputs/cuts/<cut>/final/<backbone>/best_<backbone>.pth` 存在。

### YOLO 找不到 ROI

可能是影像品質、方向、裁切範圍或 confidence threshold 造成。現在 app 使用：

```text
conf=0.25
imgsz=640
```

若常常偵測不到，可以到 `web_app/app.py` 調整 `yolo_model.predict(...)` 的 `conf`。

### 預測結果和 OOF 選擇落差很大

目前 web app 和 hierarchy test 都使用 `final/` 重新訓練後的模型權重。OOF 選擇則是根據 CV fold models。這兩者不是同一批權重，因此 OOF 排名和 final model 實際表現可能不完全一致。

這是目前訓練流程中最需要注意的地方。
