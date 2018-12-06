# 一行命令训练你的图像分类模型

![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/README_IMG/01.jpg)

## 1.项目介绍

这次给大家介绍一个很方便的训练自己图像识别模型的一个程序。可以通过一行命令实现训练自己的图像识别模型，并且训练的速度很快，效果也不错。

图像分类有三种训练方式：

*   构建一个新的模型并从头开始训练，称为**scrach**。

*   在已经训练好的模型基础上，修改模型的最后的全连接层，并重新训练全连接层称为**bottleneck**。

*   在已经训练好的模型基础上，修改模型的最后的全连接层，并重新训练全连接层同时微调模型的卷积层，称为**finetune**。

我们这次项目的名称为bottleneck，所以我们要介绍的是第二种训练图像分类的方法。使用bottleneck的方式来训练自己的数据，优点是训练速度比较快，训练周期数比较少就可以得到比较好的结果。

这次训练模型使用的代码是tensorflow官方提供的图像重新训练的程序：
[程序地址](https://github.com/tensorflow/hub/tree/master/examples/image_retraining)

该程序使用了一个已经训练好的inception-v3的模型来作为基础模型，inception-v3是使用imagenet比赛的数据集训练出来的一个可以识别1000种生活中常见物体的模型。

v3表示它的第三个版本，inception模型的使用是可以参考：
[参考地址](https://github.com/Qinbf/Tensorflow/tree/master/Tensorflow基础使用与图像识别应用/程序)

**项目所需环境：**
Python
Tensorflow


## 2.Inception的bottleneck介绍

下图为inception模型的结构图：

![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/inception.jpg)


如果把这个模型看成是一个花瓶的话，那么bottleneck瓶颈就是图中全连接FC(fully connected)的部分。

inception的模型本来是为了imagenet比赛而创建的，所以有1000个分类，全连接层有1000个输出。

我们训练自己的数据集通常没有这么多分类，而且imagenet比赛中分类也不一定是我们想要的分类。

所以模型的全连接层的部分需要全部丢弃，然后创建新的全连接层并重新训练。

比如我们要训练5个分类的图片，那么就创建一个有5个输出的全连接层来训练。

模型中的卷积层部分因为是经过大量训练得到的，所以已经具备了非常良好的特征提取的能力，所以就不需要重新训练了，可以不做修改直接使用。

**训练流程：**

 1. 先把数据集传入已经训练好的inception模型，得到FC层的前一层AveragePool层的输出，这个输出值相当于是从这张图片提取出的图像特征。
    然后把这个图像特征以文本形式保存到本地文件夹中。

 2. 根据图片类别数量重新构建全连接层，比如一共有5个类别的图片那么就会创建5个输出的全连接层。

 3. 把每张图片的特征输出传到新的全连接层中做训练，只训练全连接层。

 4. 把inception中的卷积层再加上后面训练好的全连接层就得到了新的模型，可以识别我们自己特定的数据集。


## 3.Inception模型准备

Inception-v3的模型比较大，所以我上传了百度云盘：


链接：[百度云地址](https://pan.baidu.com/s/1geSXV-VsB9-4j5X5Tgcixw)
密码:**i5jw**

在程序根目录创建一个inception_model的文件夹，然后把模型的压缩包下载到inception_model的文件夹中，然后解压。如图所示：

![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%871.png)


## 4.**项目文件介绍**

![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%872.png)


**4.1** retrain.py为训练模型的程序

**4.2** Inception_model为之前训练好的inception模型存放路径

**4.3** data为需要训练的图片存放位置
![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%873.png)



图片存放的方式必须如上图所示，一个种类的图片放在一个文件夹下面。图片文件夹的名字就是图片类别的名字，名字要用英文。

**4.4** bottleneck为inception卷积层提取出来的特征，以txt的格式保存在bottleneck文件夹中
![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%874.png)


**4.5**（**Linux和mac用户**执行**retrain.sh**可以开始训练模型；**windows用户**执行**retrain.bat**可以开始训练模型）

**retrain.bat内容如下：**
```
python retrain.py ^
bottleneck_dir bottleneck ^
how_many_training_steps 200 ^
model_dir inception_model ^
output_graph output_graph.pb ^
output_labels output_labels.txt ^
image_dir data
pause
```

**注**：
> **bottleneck_dir**：图片特征文件存放路；
> **how_many_training_steps**：训练次数；
> **model_dir**：之前训练好的inception模型存放路径；
> **output_graph**：新训练的模型存放路径和名称
> **output_labels**：生成的标签文件存放路径和名称
> **image_dir**：训练数据存放路径

**4.6** test_images为测试图片存放位置

**4.7** predict.ipynb和predict.py为新训练好的模型的测试程序，执行这两个文件可以载入训练好的新的模型预测test_images文件夹中的图片类型。


## 5.训练和测试结果

我训练了5个种类的图片：'flower','guitar','house','animal','plane'

每个种类500张，特征提取的时间会比较长，我使用mac笔记本大概要二三十分钟，训练的时间很快，不到1分钟。

训练速度比较快的原因是训练次数我设置比较少只有200次，并且只训练了全连接层。训练结束后可以看到测试集准确率：

**INFO:tensorflow:Final test accuracy = 99.6% (N=247)**

几乎达到100%的准确率！

准确率之所以可以这么高**主要原因有两个**：

**第一**是由于inception**模型本身比较优秀**，我们在一个优秀的模型上进行进一步训练所以得到的结果也很好；

**第二**是由于我使用的**图像分类难度比较低**，因为，花，吉他，房子，动物，飞机这几个种类之间的相似度非常低，所以很容易分辨。

如果是做狗的品种识别，比如狗有100个品种，因为狗与狗之间的相似度很高，那么准确率就不可能达到这么高了。

predict文件的**部分测试效果**：
![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%875.png)



![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%876.png)


![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/bottleneck/README_IMG/%E5%9B%BE%E7%89%877.png)


**写在最后的话**： 如果你也是个深度学习爱好者，或者有任何的疑问，可以加我的 **个人微信号**：**sdxxqbf** 。
