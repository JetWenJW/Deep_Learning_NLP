from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 語料：最後一句為問題，其他為回答
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

# 將語料轉換為詞頻矩陣，計算各個字詞出現的次數。
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# 生字表
words = vectorizer.get_feature_names_out()
print("Vocabulary：", words)

# 查看四句話的 BOW
print("BOW=\n", X.toarray())

# TF-IDF 轉換
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
print("TF-IDF=\n", np.around(tfidf.toarray(), 4))

# 最後一句與其他句的相似度比較
similarities = cosine_similarity(tfidf[-1], tfidf[:-1])
print("Cosine Similarities：\n", similarities)
