114 專題筆記 — 桌球揮拍數據分析模型


## 一、Introduction

桌球揮拍動作包含速度、力量、角度等多種複雜因素，過去教練評估選手能力，多依賴主觀經驗。本專題使用T-Brain過去公開的桌球揮拍資料，透過分析揮拍過程的特徵，建立機器學習模型來預測選手的：

*    Play Years（球齡）

*  Hold Racket Handed（慣用手）

*  Level（實力）

*  Gender（性別）
 
我們希望以資料科學方式，提供更客觀的選手分析與潛力判斷工具。
  
  
## 二、專題介紹

### 2.1 資料來源

   

使用T-Brain桌球揮拍資料集，包含：

*  特徵欄位(cut_point) : Ax,Ay,Az,Gx,Gy,Gz

*  額外欄位：unique_id, mode

*  預測標籤：play_years,handed,level,gender



特徵欄位為選手27次的揮拍資訊，XYZ軸各自的加速度、角速度。

   

   

### 2.2 資料前處理

#### 目標：
    
    將揮拍數據透過特徵工程將特徵標準化，併入每位選手的

*  類別欄位編碼
     

*  特徵標準化(+特徵工程)
 
 
*  訓練
  
   
### 2.3 專題流程

資料清理 → 特徵工程 → Sklearn建模 → PyTorch MLP → 模型評估 → 結果比較 → 分析與結論

   



   

## 三、Machine Learning 模型訓練
### 3.1 Sklearn 模型

   

 使用模型

*  Decision tree

*  KNeighborsClassifier
   
   
 訓練方式
 
 * play years

* handed / gender / level


 訓練結果
 
 *  Play Years

 *  Hold Racket Handed

 *  Level

 *  Gender
   

   

### 3.2 PyTorch 模型（MLP）

   

 模型架構

*  Input：14 維特徵

*  Hidden Layers：2~3 層（ReLU）

*  Output：依任務不同（1 / 多類別）

 訓練設定

*  Optimizer：Adam

*  Learning rate：0.001

*  Loss：

    - MSELoss

    - CrossEntropyLoss（分類）

   

 訓練狀況

 *  遇到問題

   

 模型結果(acc，auc，f1)

 *  Play Years

 *  Handed

 *  Level

 *  Gender

   



   

## 四、模型結果分析

   

### 4.1 Sklearn vs PyTorch比較


   

### 4.2 特徵分析

   



   

### 4.3 問題與限制

 *  PyTorch需較長時間調參

   


### 4.4 未來改進



   

## 五、結論

本專題透過Sklearn與PyTorch建立桌球揮拍分析模型，能預測選手的經驗年數、慣用手、等級與性別。