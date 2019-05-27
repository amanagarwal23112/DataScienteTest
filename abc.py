# Import relevant libraries
import pandas as pd
import numpy as np
import json
from smart_open import smart_open
import os
from scipy.sparse import csr_matrix
import itertools
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing as pp
from pandas import DataFrame

# Check the Directory
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))

# Import Json and csv files
true = pd.read_csv("y_true.csv")

def read_data():
    with smart_open('./dataset.jsonl.gz') as abc:
        dataset = [
            json.loads(line.decode('utf-8'))
            for line in abc
        ]
    return pd.DataFrame(dataset)

daltix = read_data()

# Extract the data in CSV file to understand
daltix.to_csv('clean_data.csv')

# Create a new dataset - Match the daltix ID if product id is same
matches = [] 
for name, group in daltix.groupby('PRODUCT_ID'):
    matches += list(itertools.combinations(list(group['DALTIX_ID']), 2))
matched_data = pd.DataFrame(matches,columns=['daltix_id_1','daltix_id_2'])

matched_data.to_csv('abc.csv')

# Change index as daltix id for lookup
daltix = daltix.set_index('DALTIX_ID')

# After observing the data - Every first word in name is mostly the brand name 
daltix['BRAND'] = daltix['BRAND'].fillna(daltix['NAME'].str.split(' ').str[0])

# Create a new dataset - Mapp the brand in new dataset with both daltix_id_1 and daltix_id_2
matched_data['brand_id_1'] = matched_data.daltix_id_1.apply(lambda x: daltix.loc[x,'BRAND'])
matched_data['brand_id_2'] = matched_data.daltix_id_2.apply(lambda x: daltix.loc[x,'BRAND'])
matched_data['brand_id_1'] = matched_data.brand_id_1.str.lower()
matched_data['brand_id_2'] = matched_data.brand_id_2.str.lower()

# Checking how many entries are true
matched_data['filter_brand'] = (matched_data.brand_id_1 == matched_data.brand_id_2)
first_submission = matched_data[matched_data.filter_brand == True]
print(matched_data['filter_brand'].value_counts())   # 27213= True & 3238 = False

# how many times product id i repeating
print(daltix.PRODUCT_ID.value_counts())

# Remove Dutch Stopwords
stop_words = get_stop_words('dutch')

#Create a new data again
new_daltix = read_data()

# Take only Name column 
new = new_daltix.NAME
print(type(new))

# Apply Term Frequency Inverse document frequency and cosine similarity
vectorizer = TfidfVectorizer(stop_words=stop_words)
trans = vectorizer.fit_transform(new)
# cosine_similar = trans*trans.T

# Check memoey usage of sparse matrix
def sparse_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
    except AttributeError:
        return -1

print(sparse_memory_usage(trans))  # it consumes 8090332 memory

# Reduce the memory by converting the data type from float64 to float32 and Check the memory again
trans = trans.astype(dtype='float16')

def sparse_memory_usage(mat):
    try:
        return mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes
    except AttributeError:
        return -1

print(sparse_memory_usage(trans))  # it consumes 4248730 memory
cosine_similar = trans*trans.T
print(type(cosine_similar))

# Convert sparse matrix into array
arr = np.sort(cosine_similar.toarray())

# Create a new data frame 
final = pd.DataFrame({'daltix_id_1':range(0,cosine_similar.shape[0]), 'daltix_id_2': arr[:,-2]})

# Adding a column - cosine distance in a new data frame
final['dist'] = final.apply(lambda x: cosine_similar[x['daltix_id_1'],x['daltix_id_2']] ,axis=1) 

# add new columns of brand, name, shop correspondig to daltix_id_1 & daltix_id_2   
final['daltix_id_1'] = final.daltix_id_1.apply( lambda x: new_daltix.iloc[x]['DALTIX_ID'])
final['daltix_id_2'] = final.daltix_id_2.apply( lambda x: new_daltix.iloc[x]['DALTIX_ID'])
final['brand_id_1'] = final.daltix_id_1.apply( lambda x: new_daltix.iloc[x]['BRAND'])
final['brand_id_2'] = final.daltix_id_2.apply( lambda x: new_daltix.iloc[x]['BRAND'])
final['name_id_1'] = final.daltix_id_1.apply( lambda x: new_daltix.iloc[x]['NAME'])
final['name_id_2'] = final.daltix_id_2.apply( lambda x: new_daltix.iloc[x]['NAME'])
final['shop_id_1'] = final.daltix_id_1.apply( lambda x: new_daltix.iloc[x]['SHOP'])
final['shop_id_2'] = final.daltix_id_2.apply( lambda x: new_daltix.iloc[x]['SHOP'])
final['digits_id_1'] = final.name_id_1.apply(func)
final['digits_id_2'] = final.name_id_2.apply(func)
print(final['digits_id_1'])

# filter the data with some conditions
final = final[(final.brand_id_1 == final.brand_id_2) & (final['shop_id_1'] != final['shop_id_2'])]
final = final[(final.digits_id_1 == final.digits_id_2) | (len(final.digits_id_1) == 0) | (len(final.digits_id_2) == 0)]

# Add dist column in matched_data
def vec(daltix_id):
    return trans[new_daltix.loc[daltix_id]['index']].values

matched_data['dist'] = vec(matched_data.daltix_id_1) * vec(matched_data.daltix_id_2)

# Concatenate the final data and matched data
final_analysis = pd.concat([matched_data,final[['daltix_id_1','daltix_id_2']]])

# Extract the dist>0.90
test2 = final_analysis[final_analysis['dist']>0.9]
test3 = final_analysis[final_analysis['dist']>0.8]
test4 = final_analysis[final_analysis['dist']>0.7]
test5 = final_analysis[final_analysis['dist']>0.6]
test6 = final_analysis[final_analysis['dist']>0.5]
test7 = final_analysis[final_analysis['dist']>0.4]
test8 = final_analysis[final_analysis['dist']>0.3]


# Evaluation
len_validation = len(true)
len_submission = len(test2)
tp = len(test2.intersection(true))
fp = len(test2 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test2 = 2/((1/recall) + (1/precision))
print(f1_test2)

len_validation = len(true)
len_submission = len(test3)
tp = len(test3.intersection(true))
fp = len(test3 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test3 = 2/((1/recall) + (1/precision))
print(f1_test3)

len_validation = len(true)
len_submission = len(test4)
tp = len(test4.intersection(true))
fp = len(test4 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test4 = 2/((1/recall) + (1/precision))
print(f1_test4)

len_validation = len(true)
len_submission = len(test5)
tp = len(test5.intersection(true))
fp = len(test5 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test5 = 2/((1/recall) + (1/precision))
print(f1_test5)

len_validation = len(true)
len_submission = len(test6)
tp = len(test6.intersection(true))
fp = len(test6 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test6 = 2/((1/recall) + (1/precision))
print(f1_test6)

len_validation = len(true)
len_submission = len(test7)
tp = len(test7.intersection(true))
fp = len(test7 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test7 = 2/((1/recall) + (1/precision))
print(f1_test7)

len_validation = len(true)
len_submission = len(test8)
tp = len(test8.intersection(true))
fp = len(test8 - true)
recall = tp/len_validation
precision = tp/len_submission
fpr = fp/len_submission
f1_test8 = 2/((1/recall) + (1/precision))
print(f1_test8)


# Extracting the Submission file
