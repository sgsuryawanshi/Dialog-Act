import re
import itertools
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
f = open("NLU.train", "r")
statement = []
nameTag = ""
idTag = ""
tagStart = False
finalListdummy1=[]
finalListdummy2=[]

f1 = open("NLU.test", "r")
## function to populate  IsLCap,IsRCap,LlessThan3 columns
def CoreFeaturePopulate(dataframe):
    dataframe['IsLCap']=0
    dataframe.loc[dataframe.Word.str.istitle(), 'IsLCap'] = 1
    dataframe['IsAllCap']=0
    dataframe.loc[dataframe.Word.str.isupper(), 'IsAllCap'] = 1
    dataframe['IsAllLow']=0
    dataframe.loc[dataframe.Word.str.islower(), 'IsAllLow'] = 1
    dataframe['Islength']=0
    dataframe.loc[dataframe.Word.str.len()>3, 'Islength'] = 1
    dataframe['Islength8']=0
    dataframe.loc[dataframe.Word.str.len()>8, 'Islength8'] = 1
    dataframe['IsNumber']=0
    dataframe.loc[dataframe.Word.str.isdigit(), 'IsNumber'] = 1
    dataframe['Islength2']=0
    dataframe.loc[dataframe.Word.str.len()<2, 'Islength2'] = 1
    dataframe['Isalpha']=0
    dataframe.loc[dataframe.Word.str.isalpha(), 'Isalpha'] = 1


def processText(statement,name,ids):
 finalList = []
 for eachLine in statement:
  # print("This is each Line",eachLine)
  wordList = re.sub("[^\wa-zA-Z0-9,;:\-_'\s+]", " ",eachLine).split()
  for word in wordList:
    if word in name:
      a = name.index(word)
      if a==0:
        result = word+"/B"
        finalList.append(result)
      else:
        result = word+"/I"
        finalList.append(result)

    else:
      if word in ids:
        a = ids.index(word)
        if a==0:
          result = word+"/B"
          finalList.append(result)
        else:
          result = word+"/I"
          finalList.append(result)

      else:
        result = word+"/O"
        finalList.append(result)
  #print(' '.join(finalList))
  finalListdummy1.append(finalList)
  finalList = []

for line in f:
    if "<" in line:
        tagStart = True
        continue
    else:
      # print(line)
      if tagStart == False:
        statement.append(line)
        # print("this is the line", statement)
        continue
      if "name=" in line:
        nameTag = line.split("name=",1)[1]
        continue
      if "id=" in line:
        idTag = line.split("id=",1)[1]
        continue

    if ">" in line:
        tagStart = False
        processText(statement,nameTag,idTag)
        nameTag = ""
        idTag = ""
        statement = []
        continue


def TestprocessText(statement,name,ids):
 finalList = []
 for eachLine in statement:
  # print("This is each Line",eachLine)
  wordList = re.sub("[^\wa-zA-Z0-9,;:\-_'\s+]", " ",eachLine).split()
  for word in wordList:
    if word in name:
      a = name.index(word)
      if a==0:
        result = word+"/B"
        finalList.append(result)
      else:
        result = word+"/I"
        finalList.append(result)

    else:
      if word in ids:
        a = ids.index(word)
        if a==0:
          result = word+"/B"
          finalList.append(result)
        else:
          result = word+"/I"
          finalList.append(result)

      else:
        result = word+"/O"
        finalList.append(result)
  #print(' '.join(finalList))
  finalListdummy2.append(finalList)
  finalList = []

for line in f1:
    if "<" in line:
        tagStart = True
        continue
    else:
      # print(line)
      if tagStart == False:
        statement.append(line)
        # print("this is the line", statement)
        continue
      if "name=" in line:
        nameTag = line.split("name=",1)[1]
        continue
      if "id=" in line:
        idTag = line.split("id=",1)[1]
        continue

    if ">" in line:
        tagStart = False
        TestprocessText(statement,nameTag,idTag)
        nameTag = ""
        idTag = ""
        statement = []
        continue

finalListdummy1 = [x for x in finalListdummy1 if x != []]
merged1 = list(itertools.chain.from_iterable(finalListdummy1))

finalListdummy2 = [y for y in finalListdummy2 if y != []]
merged2 = list(itertools.chain.from_iterable(finalListdummy2))


dfObj1 = pd.DataFrame(merged1)
dfObj1 = dfObj1.rename(columns={0:'WordTag'})

new1 = dfObj1["WordTag"].str.split("/", n = 1, expand = True)

dfObjtrain = new1.rename(columns={0:'Word',1:'Tag'})

CoreFeaturePopulate(dfObjtrain)


dfObj2 = pd.DataFrame(merged2)
dfObj2 = dfObj2.rename(columns={0:'WordTag'})

new2 = dfObj2["WordTag"].str.split("/", n = 1, expand = True)

dfObjtest = new2.rename(columns={0:'Word',1:'Tag'})

CoreFeaturePopulate(dfObjtest)

##preparing Xtrain and Xtest dataframe
X_train = dfObjtrain[['IsLCap','IsAllCap','IsAllLow','Islength','Islength8','Islength2','Isalpha','IsNumber']]
X_test = dfObjtest[['IsLCap','IsAllCap','IsAllLow','Islength','Islength8','Islength2','Isalpha','IsNumber']]
y_train = dfObjtrain[['Tag']]
y_test = dfObjtest[['Tag']]


model =DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)


y_predict = model.predict(X_test)

Data_File = "NLU.train"
TestsFile = "NLU.test"
outputName = "NLU.test.out"
outputFile = open(outputName, "w")


from sklearn.metrics import accuracy_score
print('Accuracy of model:',accuracy_score(y_test, y_predict))
