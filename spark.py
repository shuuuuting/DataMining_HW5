import pyspark
from pyspark import SparkContext
import numpy as np
from sklearn import metrics, tree
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score$
import pandas as pd
from pyspark.sql import SparkSession

# read files
global Path
sc = SparkContext()
if sc.master[0:5]=='local': #代表由本機執行，直接讀取本機檔案
    Path="file:///"
else: #在cluster執行，讀取hdfs檔案
    Path="hdfs://master:9000/sample/"

df = pd.read_csv(Path+'character-deaths.csv')

# fill blanks
df = df.fillna(0)

# turn num into 1
bookofdeath = np.zeros(df.shape[0])
for i in range(df.shape[0]):
    if (df['Book of Death'][i] != 0):
       bookofdeath[i] = 1
df.insert(2,'book of death',bookofdeath)

# drop Death Year, Death Chapter
df = df.drop("Book of Death", axis=1)
df = df.drop("Death Year", axis=1)
df = df.drop("Death Chapter", axis=1)

# 將Allegiances轉成dummy
dummies_1 = pd.get_dummies(df, columns=['Allegiances'], drop_first=True)
df = dummies_1

# feature target
target = df.loc[:, ['book of death']]
df_feature = df.drop("book of death", axis=1)
features = df_feature

# 亂數拆成訓練集(75%)與測試集(25%)
#x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25)
#train=pd.concat([x_train,y_train],axis=0)
msk = np.random.rand(len(df)) < 0.75
train = df[msk]
test = df[~msk]

#把前處理好的資料輸出成另外兩個csv
train.to_csv('train.csv',index=None,header=True)
test.to_csv('test.csv',index=None,header=True)

def extract_features(field,categoriesMap,featureEnd):
    #擷取分類特徵欄位
    categoryIdx = categoriesMap[field[0]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    #擷取數值欄位
    numericalFeatures=[convert_float(field)  for  field in field[1: featureEnd]]
    #回傳「分類特徵欄位」+「數字特徵欄位」
    return  np.concatenate(( categoryFeatures, numericalFeatures))

def extract_label(field):
	label=field[-1]
	return float(label)

def convert_float(x):
    return (0 if x=="?" else float(x))

train_rawDataWithHeader=sc.textFile(Path+'train.csv')
test_rawDataWithHeader=sc.textFile(Path+'test.csv')
#train_data=prepare_data(train)
header = train_rawDataWithHeader.first()
rawData = train_rawDataWithHeader.filter(lambda x: x != header)
rData = rawData.map(lambda x: x.replace("\"", ""))
train_data = rData.map(lambda x: x.split(","))
#test_data=prepare_data(test)
header = test_rawDataWithHeader.first()
rawData = test_rawDataWithHeader.filter(lambda x: x != header)
rData = rawData.map(lambda x: x.replace("\"", ""))
test_data = rData.map(lambda x: x.split(","))

lines=train_data+test_data
categoriesMap = lines.map(lambda fields: fields[0]).distinct().zipWithIndex().collectAsMap()
train_RDD=train_data.map(lambda r:LabeledPoint(extract_label(r),extract_features(r,categoriesMap,len(r)-1)))
test_RDD=test_data.map(lambda r: (extract_features(r,categoriesMap,len(r)-1),r[-1]))

model=DecisionTree.trainClassifier(train_RDD,numClasses=2,categoricalFeaturesInfo={},impurity='entropy',maxDepth=14,maxBins=9)

count = 0
num = 0
positive = 0
negative = 0
truePositive = 0
trueNegative =0
falsePositive = 0
falseNegative = 0
for data in test_RDD.take(test_data.count()):
    num+=1
    preds = int(model.predict(data[0]))
    #print(str(preds)+' '+str(data[1]))
    if(preds == int(data[1])):count=count+1
    if int(data[1]) == 0: negative += 1
    if int(data[1]) == 1: positive += 1
    if (int(preds) == 1 and int(data[1]) == 1) : truePositive +=1
    if (int(preds) == 0 and int(data[1]) == 1) : falseNegative +=1
    if (int(preds) == 0 and int(data[1]) == 0) : trueNegative +=1
    if (int(preds) == 1 and int(data[1]) == 0) : falsePositive +=1
acc = count/num
precision = float(truePositive) / (float(truePositive) + float(falsePositive))
recall =  float(truePositive) / (float(truePositive) + float(falseNegative))
print('Accuracy:'+str(acc))
print('Precision:'+str(precision))
print('Recall:'+str(recall))






