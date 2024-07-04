# 載入相關套件
import pprint  # 較美觀的列印函數
from gensim.models import Word2Vec
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities

# 語料庫
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]


# 任意設定一些停用詞
stoplist = set('for a of the and to in'.split())

# 分詞，轉小寫
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]
texts

# 單字出現次數統計
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1  
frequency     

# 移除只出現一次的單字
texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]
texts

# 轉為字典
dictionary = corpora.Dictionary(texts)

# 轉為 BOW
corpus = [dictionary.doc2bow(text) for text in texts]
corpus

# 建立 LSI (Latent semantic indexing) 模型


# num_topics=2：取二維，即兩個議題
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# 兩個議題的 LSI 公式
lsi.print_topics(2)

# 例句
doc = "Human computer interaction"

# 測試 LSI (Latent semantic indexing) 模型
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  
print(vec_lsi)

# 比較例句與語料庫每一句的相似機率

# 比較例句與語料庫的相似性索引
index = similarities.MatrixSimilarity(lsi[corpus])  

# 比較例句與語料庫的相似機率
sims = index[vec_lsi]  

# 顯示語料庫的索引值及相似機率
print(list(enumerate(sims)))  

# 依相似機率降冪排序
sims = sorted(enumerate(sims), key=lambda item: -item[1])
for doc_position, doc_score in sims:
    print(doc_score, documents[doc_position])


















