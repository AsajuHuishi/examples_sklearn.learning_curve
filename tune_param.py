# -*- coding: utf-8 -*-
import numpy as np
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.close('all')

digits = load_digits()
##查看数据集第一个数据,
#fig1 = digits['images'][0]
#plt.imshow(fig1)
#plt.title(digits['target'][0])

X = digits['data'] #(1797,64)
Y = digits['target'] #(1797,)

sgd = SGDClassifier(loss='log', shuffle=True, n_iter=5, penalty='l2', alpha=0.0001, random_state=3)

testing_range = np.logspace(-5,2,8)  #自变量
train_scores, test_scores = validation_curve(sgd, X, Y, param_name='alpha',param_range=testing_range,
                                             cv=10, scoring='accuracy', n_jobs=1)

#train_size: 5次训练的训练集的数量(5,) 
#train_scores:(5,10)每一行是10次交叉验证的得分
#test_scores:(5,10)
##对于5种不同数量的训练集，对10折交叉验证的10个训练/测试得分取平均值(即压缩列)
mean_train = np.mean(train_scores,1)  #(5,)
# 得到得分范围的上下界
upper_train = np.clip(mean_train + 0.5*np.std(train_scores,1),0,1) 
lower_train = np.clip(mean_train - 0.5*np.std(train_scores,1),0,1)
    
mean_test = np.mean(test_scores,1)
# 得到得分范围的上下界
upper_test = np.clip(mean_test + 0.5*np.std(test_scores,1),0,1) 
lower_test = np.clip(mean_test - 0.5*np.std(test_scores,1),0,1)

##画图 训练数量——训练/测试得分曲线
plt.figure('Fig2')
plt.semilogx(testing_range,mean_train,'ro-',label='train')
plt.plot(testing_range,mean_test,'go-',label='test')
##填充上下界的范围
plt.fill_between(testing_range,upper_train,lower_train,alpha=0.2,#alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明 
         color='r')                   
plt.fill_between(testing_range,upper_test,lower_test,alpha=0.2,#alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明 
         color='g')  
plt.grid()
plt.xlabel('alpha parameter')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.title('SGD')
plt.savefig('alpha_parameter.png')
plt.show()