import pandas as pd
df=pd.read_csv('sarcasm_data.csv')
features=df.iloc[:,0]
labels=df.iloc[:,1]
"""importing required libraries"""
import re
from nltk.stem.porter import PorterStemmer

"""downloading stop words"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

"""Way to find root words using porterstemmer"""
ps=PorterStemmer()

"""refining the data"""
refined_data=[]
for i in range(len(features)):
    text=re.sub('[^a-zA-Z]',' ',features[:][i])
    text=text.lower()
    text=text.split()
    text=[word for word in text if word not in set(stopwords.words('english'))]
    text=[ps.stem(word) for word in text]
    text=' '.join(text)
    print(text)
    refined_data.append(text)
df = pd.DataFrame(columns=["comment", "label"])
df["comment"] =features 
df["label"] =labels 

df.to_csv("refined.csv", index=False)    
