"""This part will open the extracted data from the files to train the model and perform 
the required prediction"""
#importing required libraries
from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)

@app.route('/')
def home():
       
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Importing the data file"""
    df=pd.read_csv('refined.csv')
    
    # Features and Labels
    features=df.iloc[:10000,0]
    labels=df.iloc[:10000,1]
    # Extract Feature With CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    features=cv.fit_transform(features.values.astype('U')).toarray()
    
    """forming train test split"""
    from sklearn.model_selection import train_test_split as TTS
    f_train,f_test,l_train,l_test=TTS(features,labels,random_state=1,test_size=0.33)
    
    #training a GaussinanNB
    del features,labels,df
    from sklearn.naive_bayes import MultinomialNB   
    MNB=MultinomialNB()    
    MNB.fit(f_train,l_train)
    
    l_pred=MNB.predict(f_test)  
    
    
    from sklearn.metrics import classification_report
    print(classification_report(l_test, l_pred))
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = MNB.predict(vect)
        return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)    