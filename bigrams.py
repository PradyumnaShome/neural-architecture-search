# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import re
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



df = pd.read_csv('/kaggle/input/cs446-fa19/train.csv')

result = df['train_error']
classify = df['arch_and_hp'][1]

layers = ['conv', 'batchnorm', 'flatten', 'leaky', 'tanh', 'softmax', 'relu', 'selu', 'dropout', 'maxpool', 'linear']

vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))

layerregex = '|'.join(layers)

document = []
print (classify)

m = {}

for i in range (0, len(df['arch_and_hp'].index)):
    text = df['arch_and_hp'][i]
    text = text.replace('leaky_relu', 'leaky')

    # print (re.findall(layerregex, text))
    text = ' '.join(re.findall(layerregex, text))
    document.append(text)
X = vectorizer.fit_transform(document)

'''
print (document)
print (vectorizer.get_feature_names())
print (X.toarray())
'''

# [sum(X.toarray()) for x in zip(*input_val)]
bigrams = [sum(x) for x in zip(*X.toarray())]

print (bigrams)
