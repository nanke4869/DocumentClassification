#-*-coding:utf-8-*-

import re
import os
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tfidf(email_list, label):
    """
    特征提取 采用tfidf
    :param email_list: 合成的总文件
    :param label: 每个文件的类别标签
    :return: dataframe 表格
    """
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1, 1))  # 将原始文档集合转换为TF-IDF功能矩阵
    vectors = vectorizer.fit_transform(email_list)  # 返回Tf-idf-weighted document-term matrix
    feature_names = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    dense = vectors.todense()  # 显示矩阵
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    df['@label'] = label

    # print(df)
    return df


if __name__ == '__main__':
    email_list = []
    label = []

    for email_type in ['baseball', 'hockey']:
        path = 'email/' + email_type
        for file_name in os.listdir(path):
            try:
                email = open(path + '/' + file_name, 'r', encoding='ISO-8859-1')
                lines = email.readlines()
            except Exception as e:
                print(path + '/' + file_name + '无法打开')
                print(e)
                continue

            # 读取每一封邮件“main body”，并存入content中
            content = ''
            for i in range(3, len(lines)):
                content += lines[i]

            # 数据预处理
            # 分词
            tokens = [word for sent in nltk.sent_tokenize(content) for word in nltk.word_tokenize(sent)]
            # 去除标点符号
            # compile 返回一个匹配对象 escape 忽视掉特殊字符含义（相当于转义，显示本身含义） string.punctuation 表示所有标点符号
            pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
            tokens = filter(None, [pattern.sub('', token) for token in tokens])
            # 去除停用词
            stop = stopwords.words('english')
            tokens = [token for token in tokens if token not in stop]
            # 移除少于3个字母的单词
            tokens = [word for word in tokens if len(word) >= 3]
            # 大写字母转小写
            tokens = [word.lower() for word in tokens]
            # 词干还原
            lmtzr = WordNetLemmatizer()
            tokens = [lmtzr.lemmatize(word) for word in tokens]
            preprocessed_text = ' '.join(tokens)

            email_list.append(preprocessed_text)

            # 数据标签
            if email_type == 'baseball':
                label.append(1)
            elif email_type == 'hockey':
                label.append(-1)

    preprocess_email = tfidf(email_list, label)

    preprocess_email.to_csv('preprocess_email.csv', index=False)
    np.save('email_smo.npy', preprocess_email)
    np.save('label_smo.npy', label)