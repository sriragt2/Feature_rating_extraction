
from flask import Flask, request, Response, render_template, jsonify
import requests
from bs4 import BeautifulSoup
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from os import listdir
import os
import string
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score, auc, precision_score, f1_score, recall_score, roc_curve
import matplotlib.pyplot as plt    
import calendar
import time    
from sklearn.naive_bayes import MultinomialNB
import datetime

app = Flask(__name__)

trainingCorpus = []
testCorpus = []
svmPickleFileName = 'svmSentimentAnalyzer.pkl'
nbPickleFileName = 'nbSentimentAnalyzer.pkl'
vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=3)
tfTransformer = TfidfTransformer(use_idf = True)
svmClassifier = svm.SVC(kernel='linear', C = 1.0,probability=True)
nbClassifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.remove('not')
lemmatizer = WordNetLemmatizer()


@app.route('/')
def home():
    
    return render_template("home.html")


@app.route('/scrap/',methods=['GET'])
def scrap():
    testCorpus.clear()
    if 'scrapURL' in request.args:
        rootURL = request.args['scrapURL']
    r=requests.get(rootURL)
    c=r.content
    soup=BeautifulSoup(c,'html.parser')
    isPaging = False
    if(soup.find('div',{'class':'pagination-warpper'}) != None):
        isPaging = True
    pages = 0
    if(isPaging):
        pages = len(soup.find('div',{'class':'pagination-warpper'}).find('ul').find_all('li')) - 2
        print(pages)
        for i in range(pages):
            url = rootURL+"?pgn="+str(i+1)
            print(url)
            r1=requests.get(url)
            c1=r1.content
            soup1=BeautifulSoup(c1,'html.parser')
            print(len(testCorpus))
            getReviewComments(soup1)
            print(len(testCorpus))
    else:
        getReviewComments(soup)
    print(len(testCorpus))

    scrapedReviewsFormatted = []
    for review,tag in testCorpus:
        scrapedReviewsFormatted.append(review+'---'+tag)
    return jsonify({"reviewsScrapped":scrapedReviewsFormatted})

@app.route('/pick/',methods=['GET'])
def pick():
    testCorpus.clear()
    if 'reviewsToPick' in request.args:
        reviewsToPick = request.args['reviewsToPick']
    #fileNamePath = '''E:/Projects/Python/Sentiment Analysis/testReviews/testReviews.txt'''
    fileNamePath  = os.path.join(app.root_path, 'testReviews', 'testReviews.txt')
    loadDoc(fileNamePath,"Test",reviewsToPick)
    print(len(testCorpus))
    testCorpusFormatted = []
    for review,tag in testCorpus:
        testCorpusFormatted.append(review+'---'+tag)
    return jsonify({"testCorpus":testCorpusFormatted})

@app.route('/svm/',methods=['GET'])
def svmModel():
    result = []
    if(len(testCorpus)==0):
        print("testCorpus is empty")
    else:
        
        result.append('Recs in Training---'+str(len(trainingCorpus)))
        result.append('Recs in Test---'+str(len(testCorpus)))
        test_X = vectorizer.transform([cleanText(review) for review,tag in testCorpus])
        test_Y = tfTransformer.transform(test_X)
        predicted_svm = svmClassifier.predict(test_Y)
        accuracy = round(np.mean(predicted_svm == [sentiment for review,sentiment in testCorpus])*100,2)
        result.append('Accuracy---'+str(round(accuracy,2)))
        
        tn, fp, fn, tp = confusion_matrix([sentiment for review,sentiment in testCorpus],predicted_svm,svmClassifier.classes_).ravel()
        result.append('True Positives, False Negatives---'+str(tp)+', '+str(fn))
        result.append('False Positives, True Negatives---'+str(fp)+', '+str(tn))
        
        cohenKappa = cohen_kappa_score([sentiment for review,sentiment in testCorpus],predicted_svm)
        result.append('Cohen Kappa Score---'+str(round(cohenKappa,2)))
        
        precision = precision_score([sentiment for review,sentiment in testCorpus],predicted_svm, average='binary',pos_label='positive')
        result.append('Precision ---'+str(round(precision,2)))
        
        f1Score = f1_score([sentiment for review,sentiment in testCorpus],predicted_svm, average='binary',pos_label='positive')
        result.append('F1 Score ---'+str(round(f1Score,2) ))
        
        recall = recall_score([sentiment for review,sentiment in testCorpus],predicted_svm, average='binary',pos_label='positive')
        result.append('Recall ---'+str(round(recall,2) ))
        
        test_X = vectorizer.transform([cleanText(review) for review,tag in testCorpus])
        test_Y = tfTransformer.transform(test_X)
        scores = svmClassifier.decision_function(test_Y)
        fpr, tpr, thresholds = roc_curve([sentiment for review,sentiment in testCorpus], scores, pos_label='positive')
        #rocCurveFile = plotROC(fpr,tpr)
        
        aucScore = auc(fpr, tpr)

        result.append('AUC---'+str(round(aucScore,2)))
        #result.append('ROC Curve---'+rocCurveFile)
        
        
        print(result)
        return jsonify({"result":result})
    
@app.route('/nb/',methods=['GET'])
def nbModel():
    result = []
    if(len(testCorpus)==0):
        print("testCorpus is empty")
    else:
        
        result.append('Recs in Training---'+str(len(trainingCorpus)))
        result.append('Recs in Test---'+str(len(testCorpus)))
        test_X = vectorizer.transform([cleanText(review) for review,tag in testCorpus])
        test_Y = tfTransformer.transform(test_X)
        predicted_nb = nbClassifier.predict(test_Y)
        accuracy = round(np.mean(predicted_nb == [sentiment for review,sentiment in testCorpus])*100,2)
        result.append('Accuracy---'+str(round(accuracy,2)))
        
        tn, fp, fn, tp = confusion_matrix([sentiment for review,sentiment in testCorpus],predicted_nb,svmClassifier.classes_).ravel()
        result.append('True Positives, False Negatives---'+str(tp)+', '+str(fn))
        result.append('False Positives, True Negatives---'+str(fp)+', '+str(tn))
        
        cohenKappa = cohen_kappa_score([sentiment for review,sentiment in testCorpus],predicted_nb)
        result.append('Cohen Kappa Score---'+str(round(cohenKappa,2)))
        
        precision = precision_score([sentiment for review,sentiment in testCorpus],predicted_nb, average='binary',pos_label='positive')
        result.append('Precision ---'+str(round(precision,2)))
        
        f1Score = f1_score([sentiment for review,sentiment in testCorpus],predicted_nb, average='binary',pos_label='positive')
        result.append('F1 Score ---'+str(round(f1Score,2) ))
        
        recall = recall_score([sentiment for review,sentiment in testCorpus],predicted_nb, average='binary',pos_label='positive')
        result.append('Recall ---'+str(round(recall,2) ))
        
        test_X = vectorizer.transform([cleanText(review) for review,tag in testCorpus])
        test_Y = tfTransformer.transform(test_X)
        scores = svmClassifier.decision_function(test_Y)
        fpr, tpr, thresholds = roc_curve([sentiment for review,sentiment in testCorpus], scores, pos_label='positive')
        #rocCurveFile = plotROC(fpr,tpr)
        
        aucScore = auc(fpr, tpr)

        result.append('AUC---'+str(round(aucScore,2)) )
        #result.append('ROC Curve---'+rocCurveFile)
        
        print(result)
        return jsonify({"result":result})
    
@app.route('/featureScore/',methods=['GET'])
def featureScore():
    featureResult = []
    if 'feature' in request.args:
        feature = request.args['feature']
    featureTestData = []
    for review,tag in testCorpus:
        for text in review.split('.'):
            if feature in text:
                featureTestData.append(tuple([text,tag]))
                
    test_X = vectorizer.transform([cleanText(review) for review,tag in featureTestData])
    test_Y = tfTransformer.transform(test_X)
    predicted_svm = svmClassifier.predict(test_Y)
    featureResult.append("No. of sentences talked about the feature '"+feature+"'---"+str(len(featureTestData)))
    featureResult.append("No. of sentences having positive sentiment ---"+str(predicted_svm.tolist().count('positive')))
    featureResult.append("Rating of the feature ---"+str(round(((predicted_svm.tolist().count('positive')/len(featureTestData))/2)*10,2))+" out of 5")
    return jsonify({"featureResult":featureResult})

@app.route('/getSentiment/',methods=['GET'])
def getSentiment():
    sentimentResult = []
    if 'text' in request.args:
        text = request.args['text']
                
    test_X = vectorizer.transform([cleanText(text)])
    test_Y = tfTransformer.transform(test_X)
    predicted_svm = svmClassifier.predict(test_Y)
    fileNamePath = ''
    if(predicted_svm[0] == 'positive'):
        fileNamePath = '/static/positive.jpg'
    else:
        fileNamePath = '/static/negative.jpg'
        
    sentimentResult.append('sentiment---'+predicted_svm[0])
    sentimentResult.append('file---'+fileNamePath)
    
    return jsonify({"sentimentResult":sentimentResult})
    
def plotROC(fpr,tpr):
    print("before fig")
    fig = plt.figure()
    print("after fig")
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    
    #timing = calendar.timegm(time.strptime('Jul 9, 2009 @ 20:02:58 UTC', '%b %d, %Y @ %H:%M:%S UTC'))
    timing = calendar.timegm(time.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
    fileName = 'rvmRoc_'+str(timing)+'.png'
    fileNamePath  = os.path.join(app.root_path, 'static', fileName)
    fig.savefig(fileNamePath , format='png')
    fileNamePath = '/static/'+fileName
    plt.gcf().clear()
    plt.close(fig)
    return fileNamePath

def getReviewComments(soup):
    propertyRowDiv = soup.find_all('div',{'class':'ebay-review-section'})
        #for propertyDiv in propertyRowDiv:
    for item in propertyRowDiv:
        ratingDiv = item.find('div',{'class':'ebay-review-section-l'})
        isPositive = False
        rating = ratingDiv.find('div',{'class':'ebay-star-rating'})
        ratingValue = int(rating.find('meta')['content'])
        if(ratingValue >= 0 and ratingValue < 3):
            isPositive = False
        elif(ratingValue > 3 and ratingValue <= 5):
            isPositive = True
        
        commentsDiv = item.find('div',{'class':'ebay-review-section-r'})
        if(ratingValue >= 0 and ratingValue <= 5 and ratingValue != 3):
            if(commentsDiv.find('p',{'class':'review-item-content'}) != None):
                comment = commentsDiv.find('p',{'class':'review-item-content'}).text
                testCorpus.append((comment,'positive' if isPositive else 'negative'))
        print(len(testCorpus))


def loadDoc(filename,dataType,recordsToPick):
    posCount =0
    negCount =0
    testCorpus.clear()
    # open the file as read only
    file = open(filename, 'r',encoding="utf8",)
    # read all text
    review = file.readline()
    count = 1
    while review:
        if(count-1==int(recordsToPick)):
            break
        else:
            count += 1
            for tag, review in [tuple(review.split(" ", 1))]:
                if(tag[-1]=='1'):
                    negCount += 1
                    tag="negative"
                elif(tag[-1]=='2'):
                    posCount += 1
                    tag="positive"
                else:
                    tag="neutral"
            if dataType=="Train":
                trainingCorpus.append((review,tag))
            elif dataType == "Test":
                testCorpus.append((review,tag))
            review = file.readline()
    file.close()

def cleanText(summary):
    summary = summary.replace('*','').replace('-',' ').replace('/',' ').replace("'",' ')
    tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split() if str not in stopwords]
    lemma_summary = [lemmatizer.lemmatize(token) for token in tokens_summary if len(token) > 0]
    #for word in lemma_summary:
    #    wordList.append(word)
    return(' '.join(word for word in lemma_summary))
    


if __name__ == "__main__":
    #fileNamePath = '''E:/Projects/Python/Sentiment Analysis/trainReviews/trainReviews.txt'''
    fileNamePath  = os.path.join(app.root_path, 'trainReviews', 'trainReviews.txt')
    
    loadDoc(fileNamePath,"Train",-1)
    
    X = vectorizer.fit_transform([cleanText(review) for review,tag in trainingCorpus])
    Y = tfTransformer.fit_transform(X)
    
    
    if(os.path.exists(svmPickleFileName) != True):
        pickle_file = open(svmPickleFileName,'wb')
        svmClassifier.fit(Y,list(sentiment for review,sentiment in trainingCorpus)) 
        pickle.dump(svmClassifier,pickle_file)
        pickle_file.close()
    else:
        unpickle_file = open(svmPickleFileName,'rb')
        svmClassifier = pickle.load(unpickle_file)
        unpickle_file.close()
    
    if(os.path.exists(nbPickleFileName) != True):
        pickle_file = open(nbPickleFileName,'wb')
        nbClassifier.fit(Y,list(sentiment for review,sentiment in trainingCorpus)) 
        pickle.dump(nbClassifier,pickle_file)
        pickle_file.close()
    else:
        unpickle_file = open(nbPickleFileName,'rb')
        nbClassifier = pickle.load(unpickle_file)
        unpickle_file.close()
    
    
    app.run(host='0.0.0.0',debug=True)
    
    
