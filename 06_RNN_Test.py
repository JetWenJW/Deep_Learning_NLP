# 載入相關套件
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 建立模型
model = tf.keras.Sequential()

# 模型只含嵌入層(Embedding layer)
# 字彙表最大為1000，輸出維度為 64，輸入的字數為 10
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# 產生亂數資料，32筆資料，每筆 10 個數字
input_array = np.random.randint(1000, size=(32, 10))

# 指定優化器、損失函數
model.compile('rmsprop', 'mse')

# 預測
output_array = model.predict(input_array)
print(output_array.shape)
output_array[0]

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 測試資料
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# 轉成 one-hot encoding
vocab_size = 50 # 字典最大字數
maxlen = 4      # 語句最大字數
encoded_docs = [one_hot(d, vocab_size) for d in docs]

# 轉成固定長度，長度不足則後面補空白
padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')

# 模型只有 Embedding
model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 64, input_length=maxlen))
model.compile('rmsprop', 'mse')

# 預測
output_array = model.predict(padded_docs)
output_array.shape

# one-hot encoding 轉換結果
print(encoded_docs[0])

# 補空白後的輸入維度
print(padded_docs.shape)

# 加上完全連接層(Dense)
# 定義 10 個語句的正面(1)或負面(0)的情緒
labels = np.array([1,1,1,1,1,0,0,0,0,0])

vocab_size = 50
maxlen = 4
encoded_docs = [one_hot(d, vocab_size) for d in docs]
padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')

model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=maxlen))
model.add(layers.Flatten())

# 加上完全連接層(Dense)
model.add(layers.Dense(1, activation='sigmoid'))

# 指定優化器、損失函數
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])

print(model.summary())

# 模型訓練
model.fit(padded_docs, labels, epochs=50, verbose=0)

# 模型評估
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

model.predict(padded_docs)


#加上 RNN 神經層
model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=maxlen))

# 加上 RNN 神經層，輸出 128 個神經元
model.add(layers.SimpleRNN(128))

# 加上完全連接層(Dense)
model.add(layers.Dense(1, activation='sigmoid'))

# 指定優化器、損失函數
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])

print(model.summary())
# 模型訓練
model.fit(padded_docs, labels, epochs=50, verbose=0)

# 模型評估
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

model.predict(padded_docs)

list(model.predict_classes(padded_docs).reshape(-1))


# load the whole embedding into memory
embeddings_index = dict()
f = open('./GloVe/glove.6B.300d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# 分詞
from tensorflow.keras.preprocessing.text import Tokenizer
t = Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1

# 轉為序列整數
encoded_docs = t.texts_to_sequences(docs)

# 補 0
padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')
padded_docs

# 轉換為 GloVe 300維的詞向量
# 初始化輸出
embedding_matrix = np.zeros((vocab_size, 300))

# 讀取詞向量值
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 任取一筆觀察        
embedding_matrix[2]


# Embedding 設為不需訓練，直接輸入轉換後的向量
model = tf.keras.Sequential()

# trainable=False：不需訓練，直接輸入轉換後的向量
model.add(layers.Embedding(vocab_size, 300, weights=[embedding_matrix], 
                           input_length=maxlen, trainable=False))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(1, activation='sigmoid'))

# 指定優化器、損失函數
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy'])

print(model.summary())

# 模型訓練
model.fit(padded_docs, labels, epochs=50, verbose=0)

# 模型評估
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

list(model.predict_classes(padded_docs).reshape(-1))

model.predict(padded_docs)