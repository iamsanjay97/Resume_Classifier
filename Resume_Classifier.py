# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:09:52 2017

@author: DELL
"""


import os,sys,re
import numpy as np
np.set_printoptions(threshold=np.nan)
from os.path import join
from nltk.corpus import stopwords,names
from nltk.stem.lancaster import LancasterStemmer
import nltk.sem.chat80 as a
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import shutil
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

def load_file():
    name = askopenfilename(filetypes=(("Text files", "*.txt") ,("HTML files", "*.html;*.htm")))        
    selection = "Selected File : " + str(name)
    label1.config(text = selection)
        
    try:
        print (name)
        
        global x
        
        f1=open(name,"r")
        x=f1.read()
        f1.close()
    except:
        T.insert(END, "NO Such File")
            
def submit():
    T.pack(side=TOP , fill = X ,padx=20 , pady = 5)
    prob_button.pack(fill=X,padx=20,pady=10)
    button.config(state = DISABLED)
    submit_button.config(state = DISABLED)
    
    if __name__=='__main__':                       
        k=news_classifier()
        
        print(var.get())
    
        if(var.get() == 1):
            k.main()
        elif(var.get() == 2):
            k.main2()
      
def reset():
    T.delete(1.0,END) 
    T2.delete(1.0,END) 
    label1.config(text = "")
    button.config(state = ACTIVE)
    submit_button.config(state = ACTIVE)
    prob_button.pack_forget()
    T.pack_forget()
    T2.pack_forget()
    var.set(1)
    
def prob(): 
    T2.pack(padx=20 , pady = 5 , expand=True, fill='both')
    #T2.pack(side=TOP , fill = X ,padx=20 , pady = 5)

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

class news_classifier():
    
    features=[]
    list=[]
    
    def tokenize(self,text):
        terms = re.findall(r'\w+', text)
        print("total words:",len(terms))
        terms = [term.lower() for term in terms if not term.isdigit()] 
        return terms
    
    def frequency(self,text):
        sentence=self.tokenize(text)
        sentence = pos_tag(sentence)
        sent=([b[0]  for b in sentence if ((b[-1] == 'JJ')or(b[-1]=='NN')or(b[-1]=='NNS')or(b[-1]=='NNP')or(b[-1]=='NNPS')or(b[-1]=='VBP')or(b[-1]=='POS')or(b[-1]=='PDT'))])
        string=""
        for i in sent:
            if i not in stopwords.words('english')+names.words()+Li+CC:
                las=LancasterStemmer()
                temp=las.stem(i)
                lemma = nltk.wordnet.WordNetLemmatizer()
                lemma.lemmatize(temp)
                string+=str(str(i)+" ")
        t = re.findall(r'\w+', string)
        print("After preprocessing total words:",len(t))
        return string
        
    def main(self):
        dataset_path=os.path.dirname(os.path.realpath(__file__))
        
        L=[]
        file=[]
        print()
        print("Data preprocesing in: ")        
        for dirname in os.listdir(dataset_path):
            print(dataset_path)
            print(dirname)
            classpath = join(dataset_path, dirname)
            print(classpath)
            for dirpath, dirnames, filenames in os.walk(classpath):
                print("Classpath:",dirpath)
                for filename in filenames:
                    print("file: ",filename,'...')
                    file.append(dirname)
                    filepath = join(dirpath, filename)
                    f=open(filepath,"r")
                    freq=self.frequency(f.read())
                    L.append(freq)
                    
                    f.close()
                    f=open("buffer1.csv","a")
                    f.write(str(freq+"\n"))
                    f.close()
                    f=open("buffer2.csv","a")
                    f.write(str(dirname+"\n"))
                    f.close()
                    print("Status: successfull")
        print("Successful data preprocessing")
        os.remove("data1.csv")
        os.remove("file1.csv")
        shutil.copyfile("buffer1.csv","data1.csv")
        shutil.copyfile("buffer2.csv","file1.csv")
        os.remove("buffer1.csv")
        os.remove("buffer2.csv")
        print("Data successfully written to data1.csv file")
        
        vectorizer = CountVectorizer(min_df=0)
        X = vectorizer.fit_transform(L)
        transformer = TfidfTransformer(smooth_idf=True)
        tfidf = transformer.fit_transform(X)
        print()
        print("TF-IDF Matrix:",tfidf.shape)
        clf = MultinomialNB()
        clf.fit(tfidf,file)
        print()
        fr=self.frequency(x)
        print()
        print("Prediction:")
        print(clf.predict_proba(vectorizer.transform([fr]).toarray()))
        pred = (clf.predict_proba(vectorizer.transform([fr]).toarray())).tolist()
        m=sorted(set(file))
        k = 0
        
        for i in pred:
            for j in i:
                T2.insert(END , (m[k] + " : " + str(j)))
                k = k + 1
                
                if(k >= len(m)):
                    break
                else:
                    T2.insert(END , "\n")
                
        print(clf.predict(vectorizer.transform([fr]).toarray()))
        
        ans = clf.predict(vectorizer.transform([fr]).toarray())
        
        T.insert(END , ("Prediction : " + str(ans[0])))
        da=[]
        datafile=open("data1.csv","r")
        data=datafile.read().splitlines()
        datafile.close()
        for f in data:
            da.append(f)
        if fr not in da:  
            t = re.findall(r'\w+', x)
            t=(t[0:10])
            a=""
            for term in t:
                a+=str(term+" ")
            s=str(dataset_path+"\\"+str(clf.predict(vectorizer.transform([fr]).toarray())[0]))
            f=open(s+"\\"+a+".txt","w")
            f.write(str(x+"\n"))
            f.close()
            f=open("data1.csv","a")
            f.write(str(fr+"\n"))
            f.close()
            f=open("file1.csv","a")
            f.write(str(clf.predict(vectorizer.transform([fr]).toarray())[0])+"\n")
            f.close()


    def main2(self):
        file=[]
        file1=open("data1.csv","r")
        corpus=file1.read().splitlines()
        file1.close()
        file1=open("file1.csv","r")
        fi=file1.read().splitlines()
        file1.close()
        for f in fi:
            file.append(f)
        import matplotlib.pyplot as plt
        print()
        print("Data successfully read from data1.csv")
        vectorizer = CountVectorizer(min_df=0)
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer(smooth_idf=True)
        tfidf = transformer.fit_transform(X)
        print()
        print("TF-IDF Matrix:",tfidf.shape)
        clf = MultinomialNB()
        clf.fit(tfidf,file)
        print()
        fr=self.frequency(x)
        print()
        print("Prediction:")
        print(clf.predict_proba(vectorizer.transform([fr]).toarray()))
        pred = (clf.predict_proba(vectorizer.transform([fr]).toarray())).tolist()
        m=sorted(set(file))

        k = 0
        for i in pred:
            for j in i:
                T2.insert(END , (m[k] + " : " + str(j)))
                k = k + 1
                
                if(k >= len(m)):
                    break
                else:
                    T2.insert(END , "\n")
            
        print(clf.predict(vectorizer.transform([fr]).toarray()))
        
        ans = clf.predict(vectorizer.transform([fr]).toarray())
        da=[]
        T.insert(END , ("Prediction : " + str(ans[0])))
        dataset_path=os.path.dirname(os.path.realpath(__file__))
        datafile=open("data1.csv","r")
        data=datafile.read().splitlines()
        datafile.close()
        for f in data:
            da.append(f)
        if fr not in da:  
            t = re.findall(r'\w+', x)
            t=(t[0:10])
            a=""
            for term in t:
                a+=str(term+" ")
            s=str(dataset_path+"\\"+str(clf.predict(vectorizer.transform([fr]).toarray())[0]))
            f=open(s+"\\"+a+".txt","w")
            f.write(str(x+"\n"))
            f.close()
            f=open("data1.csv","a")
            f.write(str(fr+"\n"))
            f.close()
            f=open("file1.csv","a")
            f.write(str(clf.predict(vectorizer.transform([fr]).toarray())[0])+"\n")
            f.close()
        

root = Tk()
root.configure(background='#ffffff')

root.minsize(width=300,height=200)
root.maxsize(width=800,height=600)

root.title("Resume Classifier")

button = Button(root, text="Choose a File", bg = "#2fa4e7" ,fg = "#ffffff", command=load_file, width=10 , font = "Helvetica 10 bold")
button.pack(fill=X,padx=20,pady=15)

label1 = Label(root , text = "Selected File : " , bg = "#ffffff" ,fg = "#000000" , font = "Ariel 8 bold")
label1.pack(fill=X,padx=20)

'''entry = Entry(master, bd =1)
entry.pack()'''

temp = Label(root , text = "-----------------------------------------------------------------------------------------------------------" , bg = "#ffffff")
temp.pack(fill = X ,padx=20 , pady = 2)    

mode = Label(root , text = "Select model testing mode: " , bg = "#ffffff" , font = "Helvetica 10 bold")
mode.pack(fill = X ,padx=20 , pady = 2)          

var = IntVar()
R1 = Radiobutton(root, text="Train the model and test a resume", bg = "#ffffff" , variable=var, value=1)
R1.pack(anchor = W , padx = 20)

R2 = Radiobutton(root, text="Test a resume",bg = "#ffffff" , variable=var, value=2)
R2.pack(anchor = W , padx = 20)

var.set(1)

submit_button = Button(root, text="Submit", bg = "#2fa4e7" ,fg = "#ffffff",command=submit , font = "Helvetica 10 bold")
submit_button.pack(fill=X,padx=20,pady=5)

reset_button = Button(root, text="Reset", bg = "#2fa4e7" ,fg = "#ffffff",command=reset , font = "Helvetica 10 bold")
reset_button.pack(fill=X,padx=20,pady=5)

T = Text(root, height=1, width=1000)
T.bind("<Key>", lambda e: "break")
T2 = ScrolledText(root, undo=True)

prob_button = Button(root, text="View Probability", bg = "#2fa4e7" ,fg = "#ffffff",command=prob , font = "Helvetica 10 bold")

root.mainloop()
