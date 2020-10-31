
import nltk 
import string 
import re
import inflect
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
class PreProcessing():
    def __init__(self,data):
        self.data = data
    def preProcess(this):
        d = this.data
        f=[]
        for row in d:
            #print(row)
            row = this.removeWhitespaces(row)
            #print(row)
            row = this.removeStopwords(row)
            #print(row)
            row = this.textLowerCase(row)
            #print(row)
            row = this.removeNumbers(row)
            #print(row)
            row = this.convertNumbers(row)
            #print(row)
            row = this.removePunctuation(row)
            #print(row)
            f.append(row)
        this.data=f
        return f
        #print(this.data)
    def removeWhitespaces(this,data):
        return " ".join(data.split())
    def removeStopwords(this,data):
        stopWords = set(stopwords.words("english")) 
        wordTokens = this.wordTokenize(data)
        filtered_text = [word for word in wordTokens if word not in stopWords]
        return " ".join(filtered_text)
    def wordTokenize(this,doc):
        return word_tokenize(doc)
    def textLowerCase(this,data):
        return data.lower()
    def removeNumbers(this,data):
        return re.sub(r'\d+', '', data) 
    def convertNumbers(this,data):
        p = inflect.engine()
        # split string into list of words 
        temp_str = data.split() 
        # initialise empty list 
        new_string = [] 
    
        for word in temp_str: 
            # if word is a digit, convert the digit 
            # to numbers and append into the new_string list 
            if word.isdigit(): 
                temp = p.number_to_words(word) 
                new_string.append(temp) 
    
            # append the word as it is 
            else: 
                new_string.append(word) 
    
        # join the words of new_string to form a string 
        temp_str = ' '.join(new_string) 
        return temp_str
    def removePunctuation(this,data):
        return data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation), ''))
    def applyStemer(this, data):
        stemer = PorterStemmer()
        return " ".join([stemer.stem(plural) for plural in data])
    
class Input():
    def readFiles(this):
        trainData = []
        testData = []
        # reads the training data
        with open('corona_data/train.tsv', 'r') as trainFile:
            for line in trainFile:
                trainData.append(line)
        
        # reads the test data
        with open('corona_data/test.tsv', 'r') as testFile:
            for line in testFile:
                testData.append(line)
        return trainData, testData
    def separateLabels(this,data):
        documents = []
        labels = []
        for line in data:
            splitted_line = line.split('\t', 2)
            # separate the labels and examples (docs) in different list
            labels.append(splitted_line[1])
            documents.append(splitted_line[2])
        return documents, labels
class Train():
    def __init__(self,documents,labels,testDocuments,testLabels):
        self.documents = documents
        self.labels = labels
        self.testDocuments = testDocuments
        self.testLabels = testLabels
    def identity(this,x):
        return x
    def vectorization(this,is_tfidf):
        if is_tfidf:
            vec = TfidfVectorizer(preprocessor = this.identity, lowercase=True, analyzer='char', 
                                tokenizer = this.identity, ngram_range=(2,5))
        else:
            vec = CountVectorizer(preprocessor = this.identity,
                                tokenizer = this.identity)
        return vec
    def evaluation_results(this,test_lbls, predict, classifier):
        print("Accuracy = ",accuracy_score(test_lbls, predict))
        print(classification_report(test_lbls, predict, labels=classifier.classes_, target_names=None, sample_weight=None, digits=3))

    def train(this,cValue,kValue):
        print("C:",end=" ")
        print(cValue)
        vec = this.vectorization(is_tfidf=True)
        classifier = Pipeline( [('vec', vec),('cls', SVC(kernel=kValue,gamma=0.7,C=cValue))])
        classifier.fit(this.documents, this.labels)
        predict = classifier.predict(this.testDocuments)
        this.evaluation_results(this.testLabels, predict, classifier)
class Main():
    def main(this):
        inp = Input()
        print('Reading The Dataset....')
        trainData, testData = inp.readFiles()

        trainDocs, train_lbls = inp.separateLabels(trainData)
        testDocs, testLbls = inp.separateLabels(testData)
        print("Preprocessing.....")
        preObject = PreProcessing(trainDocs)
        trainDocs = preObject.preProcess()

        trainObj = Train(documents=trainDocs,labels=train_lbls,testDocuments=testDocs,testLabels=testLbls)
        print("Training the model.....")
        for i in range(1,20):
            trainObj.train(i/10,'linear')
            trainObj.train(i/10,'rbf')
        
        #print(Train(testDocs,testLbls).documents)
if(__name__=='__main__'):
    obj = Main()
    obj.main()