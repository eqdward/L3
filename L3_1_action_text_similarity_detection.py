# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:33:01 2020

"""

import pandas as pd
import numpy as np
import jieba

data = pd.read_csv(r'C:\Users\yy\Desktop\BI\L3\L3_1\L3_1_code\Assignment\sqlResult.csv', encoding='gb18030')   # 加载新闻数据
print(data.shape)

news = data.dropna(subset=['content'])   # 删除content为空的样本
print(news.shape)


# 获取停用词
def get_stopwords():
    sw = set()
    stop_words_file = r'C:\Users\yy\Desktop\BI\L3\L3_1\L3_1_code\Assignment\chinese_stopwords.txt'
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        for i in f:
            sw.add(i.strip()) 
    return sw
stopwords = get_stopwords()

# 清楚无意义符号
def clear(content):
    import re
    meaningless_symbols = re.compile(r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~—!，。？、￥…（）：【】《》‘’“”\s]+")
    return meaningless_symbols.sub('', content)

# 文本内容分割，去除停用词及无意义符号
def split_content(content):
    temp = clear(content)
    temp = jieba.cut(temp)
    result = ' '.join([i for i in temp if i not in stopwords])
    return result

"""
检验分词功能
test = news['content'].iloc[0]
print(test)
test = split_content(test)
print(test)
"""

# 对新闻内容进行分词处理
corpus = list(map(split_content, [str(i) for i in news['content']]))

# 计算corpus的TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), min_df=0.015, encoding='gb18030')
tfidf = tfidf_vec.fit_transform(corpus)

# 标记是否为自己的新闻
label = list(map(lambda s: 1 if '新华社' in str(s) else 0, news['source']))

# 建模模型，实现新闻内容是否与新华社新闻风格相同的预测
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(tfidf, label, test_size = 0.3, random_state = 666)
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_predict = nb_clf.predict(X_test)

print('准确率：% .4lf' % accuracy_score(y_test, y_predict))   #准确率： 0.8772
print('精确率：% .4lf' % precision_score(y_test, y_predict))   #精确率： 0.9730
print('召回率：% .4lf' % recall_score(y_test, y_predict))   #召回率： 0.8887

# 用模型对全样本风格进行预测，找到抄袭嫌疑的新闻
prediction = nb_clf.predict(tfidf.toarray())
labels = np.array(label)
compare_news_type = pd.DataFrame({'prediction':prediction, 'labels':labels})
copy_news_index = compare_news_type[(compare_news_type['prediction']==1)&(compare_news_type['labels']==0)].index   # 抄袭嫌疑的新闻index
my_news_index = compare_news_type[(compare_news_type['labels']==1)].index   # 属于我原创的新闻index
print("可能抄袭的新闻的条数：", len(copy_news_index))

# 对可能抄袭文章进行聚类
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
normalizer = Normalizer()
normalized_tfidf = normalizer.fit_transform(tfidf.toarray())

km_cluster = KMeans(n_clusters = 25)
k_labels = km_cluster.fit_predict(normalized_tfidf)
print(k_labels.shape)

# 创建id_class，即每一个index对应的新闻所属的分类
id_class = {index:class_ for index, class_ in enumerate(k_labels)}

# 创建class_id，即将同一个类型新闻对应的id形成一个集合，并生成字典
from collections import defaultdict
class_id = defaultdict(set)
for index, class_ in id_class.items():
    # 只统计新华社的文章
    if index in my_news_index:
        class_id[class_].add(index)

# 查找相似的文章
from sklearn.metrics.pairwise import cosine_similarity
def text_similarity_detection(cpindex, top=10):
    dist_dict = {i:cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x:x[1][0], reverse = True)[:top]

cpindex = 3352
print("是否在新华社列表：", cpindex in my_news_index)
print("是否在抄袭嫌疑列表：", cpindex in copy_news_index)

similar_list = text_similarity_detection(cpindex, top=10)
print(similar_list)

print("怀疑抄袭-----------------------------------\n", news.iloc[cpindex].content)
similar_sample = similar_list[0][0]
print("相似原文-----------------------------------\n", news.iloc[similar_sample].content)
