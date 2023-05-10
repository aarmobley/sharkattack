# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:44:22 2023

@author: Aaron Mobley
""" 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk as nltk
from wordcloud import WordCloud

sharks = pd.read_csv(r"C:\Users\Aaron Mobley\Desktop\Python\attacks.csv", encoding = 'unicode_escape', engine ='python')
sharks.head()

sharks.columns.tolist()

sharks.info()

#what questions am I trying to answer with this dataset?
#does male or female get attacked by sharks more often?
#Which body part is most likely to be attacked?
#Does the type of attack impact whether the attack is fatal?


#DATA WRANGLING

#any null data or Nan?

sharks.isna().any()
sharks.dropna(inplace=True)

sharks.info()

#any duplicate values?

sharks.duplicated().sum()

#rename columns that have a space

sharks.rename(columns={"Sex ":"Sex","Species ":"Species"},inplace=True)

sharks['Species'].value_counts()


#drop unnesessary columns 

sharks2 = sharks.drop(["pdf", 'href', 'href formula', 'Case Number.1', 'Case Number.2', "Case Number"], axis = 1)

sharks2.columns.tolist()

sharks2['Year'].value_counts() 


#Percentage of shark attacks male or female

sharks2['Sex'].value_counts()

sharks2.drop(sharks2[sharks2['Sex']=='lli'].index,inplace=True)

myexplode = [0.2, 0]
labels = ["Male","Female"]
sharks2['Sex'].value_counts().head(10).plot(kind = 'pie',autopct = '%1.1f%%', explode=myexplode ,labels = labels,shadow = True)
plt.title('Percentage of shark attacks Male/Female' )
plt.axis('equal')
plt.show()

#86% percent of shark attackes victims are male


#using text analysis on the injury column we can find which body part is most frequently attacked

from nltk.tokenize import word_tokenize
sharks2['tokens'] = sharks2['Injury'].apply(word_tokenize)


#remove stopwords from column that do not add meaning
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
sharks2['tokens'] = sharks2['Injury'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

from collections import Counter

word_count = Counter()
sharks2['tokens'].apply(lambda x: word_count.update(x))
print(word_count)

#make all words lower case

words = [word.lower() for sentence in sharks2['Injury'] for word in sentence.split()]

word_count = Counter(words)

#print top most common words
print(word_count.most_common(20))

#list specific words
#count for specifc body parts

specific_words = ['arm', 'foot', 'abdomen', 'leg', 'calf', 'hand', 'toes' 'toe',
                  'ankle', 'arms', 'arm.', 'arm,', 'arm"', 'back', 'bicep', 'buttocks',
                  'elbow', 'face', 'forearm', 'groin', 'head', 'hip', 'knee', 'legs',
                  'nose', 'shin', 'shoulder', 'testicle', 'thigh', 'thighs', 'thumb',
                  'finger', 'toe', 'torso', 'wrist']

#loop through to find the count for each word listed
specific_word_counts = {word: word_count[word] for word in specific_words}
for word, count in specific_word_counts.items():
    print(f"{word}: {count}")

#leg, foot, and thigh are the most commonly bit body parts

#create wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(specific_word_counts)

#plot wordcloud
plt.figure(figsize = (12, 6))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show



#does the activty during attack influence whether or not the attack was fatal?

sharks2['Activity'].value_counts().head(15)
sharks2['Fatal (Y/N)'].value_counts()

sharks2.to_csv(r'C:\Users\Aaron Mobley\Desktop\Python\sharks2.csv')

#using 'Type or 'Activiy as the IV?


sharks2['Type'].value_counts()
sharks2['Activity'].value_counts(10)

#remove extra space in ' N' in 'Fatal column
sharks2['Fatal (Y/N)'] = sharks2['Fatal (Y/N)'].str.strip()
sharks2['Fatal (Y/N)'].value_counts()
