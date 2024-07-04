# 載入相關套件
import gzip
import gensim 

from gensim.test.utils import common_texts
# size：詞向量的大小，window：考慮上下文各自的長度
# min_count：單字至少出現的次數，workers：執行緒個數
model_simple = gensim.models.Word2Vec(sentences=common_texts, window=1, 
                                      min_count=1, workers=4)
# 傳回 有效的字數及總處理字數
model_simple.train([["hello", "world", "michael"]], total_examples=1, epochs=2)

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

model_simple = gensim.models.Word2Vec(min_count=1)
model_simple.build_vocab(sentences)  # 建立生字表(vocabulary)
model_simple.train(sentences, total_examples=model_simple.corpus_count
                   , epochs=model_simple.epochs)

model_simple.corpus_count

model_simple.epochs

# 載入 OpinRank 語料庫：關於車輛與旅館的評論
data_file="./Word2Vec/reviews_data.txt.gz"

with gzip.open (data_file, 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break

# 讀取 OpinRank 語料庫，並作前置處理
def read_input(input_file):
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 
            # 前置處理
            yield gensim.utils.simple_preprocess(line)

# 載入 OpinRank 語料庫，分詞
documents = list(read_input(data_file))
documents

len(documents)


# Word2Vec 模型訓練，約10分鐘
model = gensim.models.Word2Vec(documents, size=150, window=10, 
                               min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


# 測試『骯髒』相似詞
w1 = "dirty"
model.wv.most_similar(positive=w1) # positive：相似詞

# 測試『禮貌』相似詞
w1 = ["polite"]
model.wv.most_similar (positive=w1, topn=6) # topn：只列出前 n 名

# 測試『法國』相似詞
w1 = ["france"]
model.wv.most_similar (positive=w1, topn=6) # topn：只列出前 n 名

# 測試『床、床單、枕頭』相似詞及『長椅』相反詞
w1 = ["bed",'sheet','pillow']
w2 = ['couch']
model.wv.most_similar (positive=w1, negative=w2, topn=10) # negative：相反詞

# 比較兩詞相似機率
model.wv.similarity(w1="dirty",w2="smelly")

model.wv.similarity(w1="dirty",w2="dirty") 

model.wv.similarity(w1="dirty",w2="clean")

# 選出較不相似的字詞
model.wv.doesnt_match(["cat","dog","france"])

# 關鍵詞萃取(Keyword Extraction)
# https://radimrehurek.com/gensim_3.8.3/summarization/keywords.html
from gensim.summarization import keywords

# 測試語料
text = '''Challenges in natural language processing frequently involve
speech recognition, natural language understanding, natural language
generation (frequently from formal, machine-readable logical forms),
connecting language and machine perception, dialog systems, or some
combination thereof.'''

# 關鍵詞萃取
print(''.join(keywords(text)))

# 下載預先訓練的模型
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

# 載入本機的預先訓練模型
from gensim.models import KeyedVectors

# 每個詞向量有 300 個元素
model = KeyedVectors.load_word2vec_format(
    './Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)

# 取得 dog 的詞向量(300個元素)
model['dog']

len(model['dog'])

# 測試『woman, king』相似詞及『man』相反詞
model.most_similar(positive=['woman', 'king'], negative=['man'])

# 選出較不相似的字詞
model.doesnt_match("breakfast cereal dinner lunch".split())

# 比較兩詞相似機率
model.similarity('woman', 'man')


import numpy as np
import nltk
import gensim
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# 測試語料
f = open('./FAQ/starbucks_faq.txt', 'r', encoding='utf8')
corpus = f.readlines()
# print(corpus)

# 參數設定
MAX_WORDS_A_LINE = 30  # 每行最多字數

# 標點符號(Punctuation)
import string
print('標點符號:', string.punctuation)

# 讀取停用詞
stopword_list = set(nltk.corpus.stopwords.words('english') 
                    + list(string.punctuation) + ['\n'])


# 分詞函數
def tokenize(text, stopwords, max_len = MAX_WORDS_A_LINE):
    return [token for token in gensim.utils.simple_preprocess(text
                              , max_len=max_len) if token not in stopwords]

# 分詞
document_tokens=[] # 整理後的字詞
for line in corpus:
    document_tokens.append(tokenize(line, stopword_list))
    
# 設定為 Gensim 標籤文件格式    
tagged_corpus = [TaggedDocument(doc, [i]) for i, doc in 
                 enumerate(document_tokens)]

# 訓練 Doc2Vec 模型
model_d2v = Doc2Vec(tagged_corpus, vector_size=MAX_WORDS_A_LINE, epochs=200)
model_d2v.train(tagged_corpus, total_examples=model_d2v.corpus_count, 
                epochs=model_d2v.epochs)

# 測試
questions = []
for i in range(len(document_tokens)):
    questions.append(model_d2v.infer_vector(document_tokens[i]))
questions = np.array(questions)    
# print(questions.shape)

# 測試語句
# text = "find allergen information"
text = "mobile pay"
filtered_tokens = tokenize(text, stopword_list) 
# print(filtered_tokens)

# 比較語句相似度
similarity = cosine_similarity(model_d2v.infer_vector(
    filtered_tokens).reshape(1, -1), questions, dense_output=False)

# 選出前 10 名
top_n = np.argsort(np.array(similarity[0]))[::-1][:10]
print(f'前 10 名 index:{top_n}\n')
for i in top_n:
    print(round(similarity[0][i], 4), corpus[i].rstrip('\n'))

