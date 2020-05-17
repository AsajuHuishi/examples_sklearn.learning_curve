## Examples of sklearn.learning_curve
These examples are applications on functions from sklearn.learning_curve module. It is based on 

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

* train number-size.png
<div align='center'>
<img src="https://img-blog.csdnimg.cn/20200506135707607.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2OTM3Njg0,size_16,color_FFFFFF,t_70">
</div>