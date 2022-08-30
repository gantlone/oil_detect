# oil_detect
use machine learning to detect oil normal and abnormal data

## 使用電腦設備
- 顯示卡: GeForce GTX 960

## 主要需求
- Matlab R2020b
- 僅用CPU也可以執行，使用GPU會更快。

## 使用流程

### 資料集來源
- 使用學長們所製作的類比數位轉換器(震動感應器)，收集中油抽油水馬達上三種震動訊號，分別為正常、鬆螺絲、偏心訊號，其中鬆螺絲與偏心訊號為異常訊號。取樣率977Hz。
- 數據筆數(一幀:5000點):
  - 正常: 358筆
  - 鬆螺絲: 546筆
  - 偏心: 386筆

### 流程圖

![流程圖]

### 步驟
- 預處理
  - 根據觀察抽油水馬達發現，開機時電壓會處於1.5V以下的震動，所以是不必要的，先行將1.5V以下訊號去除。
  - 再來使用低通濾波器，將200Hz以上的訊號濾除，其餘訊號做為保留。  
- 分幀
  - 由於訊號來源一次收集時間為15分鐘，會導致訊號取樣點過大，為80~100萬個取樣點，所以需要將訊號做分割成5000點為一幀。
  - 並設置重疊比例50%，確保每幀之間都保有連續性，也好增加學習準確度。
  - 下面左圖為分幀流程圖，右圖為三種類別分幀後的結果。

![分幀]
![分幀2]

- 特徵處理
  - 從分幀訊號上來看還是不好看出三者的差異。所以需要將訊號轉化成電腦看得懂的格式。
  - 使用時域與頻域分析，一共27種特徵處理方式。 
    - 平均值
    - 平均偏差
    - 標準差
    - 第一四分位數
    - 第三四分位數
    - 四分位距
    - 中位數
    - 峰度
    - 偏度
    - Shannon Entropy
    - Spectral Entropy
    - MFCC 1~12 
    - MFCC 訊號框
    - 功率譜最大值頻率
    - 功率譜最大值
    - 功率譜最大百分比
  - 最後，將所有特徵使用mapminmax函數，進行歸一化至(-1~1)。有些過小的特徵不被重視，使用這方法將全部特徵數值調整到大小一致。
    
- K-fold
  - 將每幀的特徵(feature)以及與其對應的標籤(label)，進行交叉驗證，方法是輪流將分為K組資料，其中一組資料當測試集，剩下K-1組為訓練集。
  - 本文K設為5，也就是說訓練測試比例為4:1，並且重複5次的循環，將五次結果做平均，即可得到整體準確度。
  
![k-fold]    

- 機器學習
  - 支持向量機(Support Vector Machine): 本研究採用Matlab中，多類支持向量機分析，函數為fitcecoc()，是一種常見的MvM技術(many vs. many)。
  - 屬於監督式學習，所以前面會將所有特徵(feature)標記對應的標籤(label)。
  - 利用核函數(Kernal Function)中的多項式參數(polynomial)，將所有特徵投放至高維空間中，將數據進行分類預測。
  
## 實驗結果

### 準確度
- 訓練準確度: 99.7%
  - 正常: 99.6%
  - 鬆螺絲: 100%
  - 偏心: 99.5%
- 測試準確度: 94.2%
  - 正常: 90.3%
  - 鬆螺絲: 98.9%
  - 偏心: 91.1%
  
![準確度]

## 結論
- 目前僅是使用機器學習的方法，結果還算不錯，都有90%以上準確度，但還沒到最好，未來可能使用深度學習或其他技巧提升整體準確度。
- 這也是我在碩士生涯中，學習機器學習的其中一個環節，真的蠻有趣的，未來也可以結合python中的keras，或許能好使用。
- 測試中還是沒有訓練上那麼好，有點過擬合，主要原因是正常訊號和偏心訊號真的太像了，這部分或許要修改特徵處理方式，也可能會更好。

## 參考來源
-  王淑麗，王朝民，黃玠華，李孟鴻，李彥賢，謝金龍，李英齊，(2020)，AI 圖像辨識應用於轉動機械電流頻譜預知保養-硫磺工場壓縮機案例，石油季刊，第56卷，第4期，pp.31-42
- D. Y. Oh, I. D. Yun, “Residual Error Based Anomaly Detection Using Auto-Encoder in SMD Machine Sound,” Sensors (Basel, Switzerland), vol.18, no.5, 1308.
- Matlab使用梅爾頻譜係數與時域分析於訊號上的應用, [Online] Available: https://www.mathworks.com/matlabcentral/fileexchange/65286-heart-sound-classifier
- H. Ling, C. Qian, W Kang, C. Liang, H. Chen, “Combination of Support Vector Machine and K-Fold cross validation to predict compressive strength of concrete in marine environment”, Construction and Building Materials, vol.206, pp.355-363, May. 2019.
- D. R. Sawitri, D. A. Asfani, M. H. Purnomo, I. K. E. Purnama and M. Ashari, “Early detection of unbalance voltage in three phase induction motor based on SVM,” 2013 9th IEEE International Symposium on Diagnostics for Electric Machines, Power Electronics and Drives (SDEMPED), 2013, pp. 573-578.
- Matlab為支持向量機或其他分類器擬合多類模型, [Online] Available: https://www.mathworks.com/help/stats/fitcecoc.html
- Matlab使用自定義的標準特徵選擇方法, [Online] Available: ttps://www.mathworks.com/help/stats/sequentialfs.html






[流程圖]:/picture/流程圖.png
[分幀]:/picture/分幀.png
[分幀2]:/picture/分幀2.png
[k-fold]:/picture/k-fold.png
[準確度]:/picture/準確度.jpg
