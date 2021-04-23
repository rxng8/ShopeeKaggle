from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np 


corpus = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document',
]

def IDF(corpus, unique_words):
    idf_dict={}
    N=len(corpus)
    for i in unique_words:
        count=0
        for sen in corpus:
            if i in sen.split():
                count=count+1
            idf_dict[i]=(math.log((1+N)/(count+1)))+1
    return idf_dict 

def fit(whole_data):
    unique_words = set()
    if isinstance(whole_data, (list,)):
        for x in whole_data:
            for y in x.split():
                if len(y)<2:
                    continue
                unique_words.add(y)
        unique_words = sorted(list(unique_words))
        vocab = {j:i for i,j in enumerate(unique_words)}
        Idf_values_of_all_unique_words=IDF(whole_data,unique_words)
    return vocab, Idf_values_of_all_unique_words
Vocabulary, idf_of_vocabulary=fit(corpus) 


print(list(Vocabulary.keys())) 

def transform(dataset,vocabulary,idf_values):
    sparse_matrix= csr_matrix( (len(dataset), len(vocabulary)), dtype=np.float64)
    for row  in range(0,len(dataset)):
        number_of_words_in_sentence=Counter(dataset[row].split())
        for word in dataset[row].split():
            if word in  list(vocabulary.keys()):
               tf_idf_value=(number_of_words_in_sentence[word]/len(dataset[row].split()))*(idf_values[word])
               sparse_matrix[row,vocabulary[word]]=tf_idf_value
    print("NORM FORM\n",normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False))
    output =normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
    return output
final_output=transform(corpus,Vocabulary,idf_of_vocabulary)
print(final_output.shape)

print(final_output[0].toarray())