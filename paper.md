114 專題筆記 — 桌球揮拍數據分析模型


## 一、Introduction

桌球揮拍動作包含速度、力量、角度等多種複雜因素，過去教練評估選手能力，多依賴主觀經驗。本專題使用T-Brain過去公開的桌球揮拍資料，透過分析揮拍過程的特徵，建立機器學習模型來預測選手的：

*    Play Years（球齡）

*  Hold Racket Handed（慣用手）

*  Level（實力）

*  Gender（性別）
 
我們希望以資料科學方式，提供更客觀的選手分析與潛力判斷工具。

  
> Repo（GitHub）：{%preview https://github.com/nineday94611/11211210 %}
## 二、專題介紹

使用T-Brain桌球揮拍資料集，包含：

- `train_info.csv` / `test_info.csv`
  - 欄位：`unique_id`, `cut_point`, `mode`（動作模式）
  - train 另外包含標籤：`play years`, `hold racket handed`, `level`, `gender`
- `train_data/{unique_id}.txt` / `test_data/{unique_id}.txt`

  - 每個 `unique_id` 代表一次測驗，包含 27 次揮拍以及前後擾動訊號，由左至右依序為:
    X軸加速度(Ax)
    Y軸加速度(Ay)
    Z軸加速度(Az)
    X軸角速度(Gx)
    Y軸角速度(Gy)
    Z軸角速度(Gz)
- `cut_point`：用來區分每次揮拍的資料（segments）


- `gender`：1=男，2=女
- `hold racket handed`：1=右手，2=左手
- `play years`：共3個球齡層(根據所有選手的球齡分布，分為 0:低、1:中、2:高)

- `level`：共4個等級(2:大專甲組選手、3:大專乙組選手、4:青少年國手、5:青少年選手)



  
  
   
### 專題流程

特徵工程 → Sklearn建模 → PyTorch MLP → 模型評估 → 結果比較 → 分析與結論

   



   

## 三、Machine Learning 模型訓練
### 3.1 特徵工程
對每個 `unique_id`：
1. 讀取 `Ax Ay Az Gx Gy Gz`
2. `cut_point` 將資料切成27段 segment
3. 每段計算統計特徵：
   - 6 軸 mean（6）
   - 6 軸 std（6）
   - 共 **12 維特徵**

### 3.2 Sklearn：Decision Tree / KNN
- 訓練資料：每個 segment 都是一筆樣本（12 維特徵）
- KNN 另外使用 `StandardScaler`（距離模型需要標準化）

### 3.3 PyTorch：MLP
對同一 `unique_id` 的所有 segments（每段 12 維）做 pooling：
- mean pooling（12 維） + max pooling（12 維）→ **24 維特徵**
再丟進 MLP，並用 4 個標籤同時預測：
- play_years（3 類）、handed（2 類）、level（4 類）、gender（2 類）
Loss 使用四個 CrossEntropy 加總。

---

   


 

   

## 四、模型結果分析

   
### 4.1 Sklearn

#### 4.1-1  Decision Tree
| Task | Acc | F1| AUC 
|---|---:|---:|---:|
| play_years  | 0.4300 | 0.3372  |-
| handed | 0.9923| 0.9924 | 0.9917
| level | 0.4643 | 0.3904| -
| gender | 0.7923 | 0.7482 | 0.5800

#### 4.1-2  KNN
| Task | Acc | F1| AUC 
|---|---:|---:|---:|
| play_years  | 0.3958 | 0.3020  |-|
| handed | 0.9965 | 0.9964 | 0.9884| 
| level | 0.4490 | 0.3666| -| 
| gender | 0.7916 | 0.7606 | 0.6078|
### 4.2 PyTorch MLP（Train / Test）

| Task | Train Acc | Test Acc | Train F1 | Test F1 |Train AUC|Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| play_years  | 0.8425 | 0.3979  |  0.8432| 0.3459 | - |-|
| handed | 0.9990 | 0.9895 | 0.9990| 0.9896 | 0.9900 |0.9600|
| level | 0.8445 | 0.4196| 0.8442 | 0.3611 | - |-|
| gender | 0.9351 | 0.7490 | 0.9349 | 0.7414 | 0.8500 |0.5600 |
### 4.3 結果分析
1. **handed 任務表現最佳**  
   三種模型在 handed 都接近 1.0，應該是慣用手造成揮拍方向與角速度特徵差異明顯。
2. **play_years 與 level 預測準確度不佳**  
   兩者在測試集約 0.40~0.47，應該是兩者都是多類欄位(`years`三類、`level`四類)，因此各類差異更複雜，需要更多特徵(可能與`mode`有關)。
3. **MLP  train 與 test 有落差**  
   MLP 訓練集表現高但測試集下降，可能為模型過擬合，未來可針對過擬合的狀況來改進模型。


   



   

## 五、問題與限制

 *  `mode` 代表不同揮拍動作型態（正/反手或不同揮拍方式）。理論上對模型訓練產生關鍵影響，但因本次使用模型的限制，若將其納入會變得過於複雜，因此本次專題中，`mode`一樣代表不同揮拍動作型態，但並沒有針對不同的形態去做特別訓練。
* 未來要改進本專題會優先針對不同`mode`去做改善。
   



   

## 六、結論

本專題使用桌球揮拍 6 軸感測訊號，透過切段與統計特徵工程建立 Sklearn 與 PyTorch 模型，完成對選手球齡、慣用手、等級與性別之預測比較。結果顯示慣用手辨識最穩定；球齡與等級仍需更細緻特徵(`mode`)以提升精確度。