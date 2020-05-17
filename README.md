## Examples of sklearn.learning_curve
These examples are applications on functions from sklearn.learning_curve module. It is based on [Python机器学习入门: sklearn.learning_curve 训练结果可视化实例（完整代码）](https://blog.csdn.net/qq_36937684/article/details/105948980) and [Python机器学习可视化（二）sklearn.validation_curve选择超参数实例（完整代码）](https://blog.csdn.net/qq_36937684/article/details/106168001).

|file name| use function|detail|
|--|--|--|
| tune_param.py |  validation_curve|SGD classifier|
| tune_train_size.py |  learning_curve|KNC classifier|

## Requirements
* Python
* sklearn
* matplotlib.pyplot

## Results
|file name| results|
|--|--|
| tune_param.py |  alpha_parameter.png|
| tune_train_size.py |  train number-size.png|

* alpha_parameter.png
<div align='center'>
<img src="https://img-blog.csdnimg.cn/20200517002306308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70">
</div>

* train number-size.png
<div align='center'>
<img src="https://img-blog.csdnimg.cn/20200506135707607.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70">
</div>
