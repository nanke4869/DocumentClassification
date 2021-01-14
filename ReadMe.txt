Case Study II: Document Classification

文件目录：
  email
  --baseball
  ----...
  --hockey
  ----...
  install_library.py：安装停用词等library程序
  email_preprocess.py：数据预处理程序
  classification_svm.py：SVM分类程序
  classification_smo.py：SMO分类程序
  preprocess_email.csv
  email_smo.npy：邮件特征二进制文件
  label_smo.npy：标签二进制文件
  report_svm.txt：5折SVM训练结果与评估
  report_smo.txt：smo的alpha和b值结果
  ReadMe.txt

步骤：
  1. 安装library
    下载‘stopwords’、‘wordnet’和‘punkt’
  2. 数据预处理
    2.1 读取email的main body部分
    2.2 分词
    2.3 去除标点符号
    2.4 去除停用词
    2.5 移除少于3个字母的单词
    2.6 大写字母转小写
    2.7 词干还原
    2.8 采用TF-IDF进行特征提取
  3. 使用SVM进行分类
    3.1 将数据处理成5折
    3.2 调用sklearn-learn中的SVM进行邮件分类
    3.3 计算每一折的precision，recall和f1值以及平均值
  4. 完成SMO算法

程序执行方法：
  1. 安装library
	python install_library.py
  2. 数据预处理
	python email_preprocess.py
  3. 使用SVM进行分类
	python classification_svm.py > report_svm.txt
  4. 完成SMO算法
	python classification_smo.py > report_smo.txt