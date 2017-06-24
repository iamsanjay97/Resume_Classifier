
import os,sys,re,csv
from os.path import join
from nltk.corpus import stopwords,names
from nltk.stem.lancaster import LancasterStemmer
import nltk.sem.chat80 as a
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import io,sparse

q = "SELECT City FROM city_table"
CC=[]
for answer in a.sql_query('corpora/city_database/city.db', q):
    CC.append(("%-10s" % answer).strip())
q = "SELECT country FROM city_table"
for answer in a.sql_query('corpora/city_database/city.db', q):
    CC.append(("%-10s" % answer).strip())
file=open("IndianPeopleSorted.csv",encoding="utf8")
corpus = file.read().splitlines()
Li=[]
for name in corpus:
    Li.extend(name.strip().split())
file.close()
file=open("cities.csv",encoding="utf8")
corpus = file.read().splitlines()
Li1=[]
for name in corpus:
    Li1.extend(name.strip().split())
file.close()
class news_classifier():
    
    features=[]
    list=[]
    #print(stopwords.words()+names.words())
    def tokenize(self,text):
        terms = re.findall(r'\w+', text) 
        terms = [term for term in terms if not term.isdigit()] 
        #print(terms)
        return terms

    def frequency(self,text):
        sent=self.tokenize(text)
        string=""
        for i in sent:
            if i not in stopwords.words('english')+names.words()+Li+CC+Li1:
                las=LancasterStemmer()
                temp=las.stem(i)
                lemma = nltk.wordnet.WordNetLemmatizer()
                lemma.lemmatize(temp)
                string+=str(temp+" ")
                    
        return string
        
    def main(self):
        dataset_path=os.path.dirname(os.path.realpath(__file__))
        #print(os.listdir(dataset_path))
        L=[]
        file=[]
        
        f=open("data1.csv","w")
        f.write("")
        f.close()
        print()
        print("Data preprocesing in: ")        
        for dirname in os.listdir(dataset_path):
            classpath = join(dataset_path, dirname)
            for dirpath, dirnames, filenames in os.walk(classpath):
                for filename in filenames:
                    print(filename,'...')
                    file.append(dirname)
                    filepath = join(dirpath, filename)
                    f=open(filepath,"r")
                    freq=self.frequency(f.read())
                    L.append(freq)
                    f.close()
                    f=open("data1.csv","a")
                    f.write(str(freq+"\n"))
                    f.close()
                    print("successfull data preprocessing in ",filename)
        print("Successful data preprocessing")
        print("Data successfully written to data1.csv file")
        vectorizer = CountVectorizer(min_df=0)
        X = vectorizer.fit_transform(L)
        #print (type(X))
        #print (X.shape)
        #print(X.toarray())
        transformer = TfidfTransformer(smooth_idf=True)
        tfidf = transformer.fit_transform(X)
        print()
        print("TF-IDF Matrix:")
        print(tfidf.toarray())
        clf = MultinomialNB()
        clf.fit(tfidf,file)
        print()
        print("Classification types:")
        print(file)
        f1=open("test1.txt","r")
        fr=self.frequency(f1.read())
        f1.close()
        print()
        print("Prediction:")
        print(str(clf.predict(vectorizer.transform([fr]).toarray())))
        
    def main2(self):
        dataset_path=os.path.dirname(os.path.realpath(__file__))
        #print(os.listdir(dataset_path))
        L=[]
        file=[]
        for dirname in os.listdir(dataset_path):
            classpath = join(dataset_path, dirname)
            for dirpath, dirnames, filenames in os.walk(classpath):
                for filename in filenames:
                    print(filename)
                    file.append(dirname)
                    filepath = join(dirpath, filename)
        
        file1=open("data1.csv","r")
        corpus=file1.read().splitlines()
        file1.close()
        print()
        print("Data successfully read from data1.csv")
        vectorizer = CountVectorizer(min_df=0)
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer(smooth_idf=True)
        tfidf = transformer.fit_transform(X)
        print()
        print("TF-IDF Matrix:")
        print(tfidf.toarray())
        clf = MultinomialNB()
        clf.fit(tfidf,file)
        print()
        print("Classification types:")
        print(file)
        f1=open("test1.txt","r")
        fr=self.frequency(f1.read())
        f1.close()
        print()
        print("Prediction:")
        print(clf.predict(vectorizer.transform([fr]).toarray()))
        
if __name__=='__main__':                       
    k=news_classifier()
    if(sys.argv[1]=="1"):
        k.main()
        #k.main2()
    else:
        if(sys.argv[1]=="2"):
            k.main2()

