# 載入相關套件
import numpy as np

# 載入GloVe詞向量檔 glove.6B.300d.txt
embeddings_dict = {}
with open("./glove/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


# 隨意測試一個單字(love)，取得 GloVe 的詞向量
embeddings_dict['love']

# 字數
len(embeddings_dict.keys())

# 以歐基里德(euclidean)距離計算相似性
from scipy.spatial.distance import euclidean

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), 
                  key=lambda word: euclidean(embeddings_dict[word], embedding))

print(find_closest_embeddings(embeddings_dict["king"])[1:10])

# 任意選 100 個單字
words =  list(embeddings_dict.keys())[100:200]
# print(words)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 以 T-SNE 降維至二個特徵
tsne = TSNE(n_components=2)
vectors = [embeddings_dict[word] for word in words]
Y = tsne.fit_transform(vectors)

# 繪製散佈圖，觀察單字相似度
plt.figure(figsize=(12, 8))
plt.axis('off')
plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")

# 測試語料：最後一句為問題，其他為回答
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

import nltk

# 參數設定
MAX_WORDS_A_LINE = 7  # 每行最多字數

# 分詞
document_tokens=[] # 整理後的字詞
token_count_per_line = [] # 每行字數
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') # 篩選文數字(Alphanumeric)
for line in corpus:
    tokens = tokenizer.tokenize(line.lower())
    token_count_per_line.append(len(tokens))
    tokens += [''] * (MAX_WORDS_A_LINE - len(tokens))
    document_tokens.append(tokens)
    
document_tokens = np.array(document_tokens, dtype=object).reshape(len(corpus), -1)    
document_tokens.shape

# 轉換詞向量
document_word_embeddings=np.zeros((len(corpus),MAX_WORDS_A_LINE,300))
for i in range(document_tokens.shape[0]):
    for j in range(document_tokens.shape[1]):
        if document_tokens[i, j] == '':
            continue
        document_word_embeddings[i, j] = embeddings_dict[document_tokens[i, j]]
    
document_word_embeddings.shape

token_count_per_line

document_word_embeddings.shape[1]


# 將同一句的每個單字詞向量平均
# 將補0的向量移除：先將之變為 nan，再使用 nanmean
for i in range(document_word_embeddings.shape[0]):
    document_word_embeddings[:, token_count_per_line[i]:, :] = np.nan
sum__word_embeddings = np.nanmean(document_word_embeddings, axis=1)
sum__word_embeddings.shape


# 字句的相似度比較
from sklearn.metrics.pairwise import cosine_similarity
print (cosine_similarity(sum__word_embeddings[-1:], sum__word_embeddings[:-1], dense_output=False))