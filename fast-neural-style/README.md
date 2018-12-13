# 一行命令对你的图像视频进行风格迁移

![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/README_IMG/01.jpg)

## 1.项目介绍
今天我们要做的是一个快速图像风格迁移的程序。

那么，什么是图像风格迁移？图像风格迁移就是把一种图像风格转变为另一种图像风格。例如，原图为：
![上海外滩](https://upload-images.jianshu.io/upload_images/14843052-0fe9ee6a5bb1f9c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

加上不同风格的图像可以得到如下不同的结果：

| configuration | style | sample |
| :---: | :----: | :----: |
| [wave.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/wave.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/wave.jpg)  |
| [cubist.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/cubist.yml) |![](https://github.com/Qinbf/tf-model-zoo/blob/master/fast-neural-style/img/style/cubist.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/cubist.jpg)  |
| [denoised_starry.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/denoised_starry.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/denoised_starry.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/denoised_starry.jpg)  |
| [mosaic.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/mosaic.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/mosaic.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/mosaic.jpg)  |
| [scream.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/scream.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/scream.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/scream.jpg)  |
| [feathers.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/feathers.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/feathers.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/feathers.jpg)  |
| [udnie.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/udnie.yml) |![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/style/udnie.jpg)|  ![](https://raw.githubusercontent.com/Qinbf/tf-model-zoo/master/fast-neural-style/img/results/udnie.jpg)  |


## 2.使用训练好的模型来生成图像

#### 2.1环境
Python
Tensorflow

#### 2.2模型下载
训练好的模型有7个，表示7种类型的风格，模型文件的百度云：
[模型的百度云地址](https://pan.baidu.com/s/1i5CioT2PUdWikUI-Y16O8w)
密码:35pg

#### 2.3使用训练好的模型
在项目根目录下执行：
```
python eval.py --model_file <your path to wave.ckpt-done> --image_file img/test.jpg
```
--model_file 是模型的路径，可以选择7个模型中的一个
--image_file是原始图片的路径

新的图片会存放在项目根目录下：generated/res.jpg


## 3.训练一个新的模型

#### 3.1下载VGG16模型
如果要训练一种新的图像风格，可以先下载VGG16的模型：
[VGG16模型](https://pan.baidu.com/s/13bJrr0PY6KhA2715SymiGw )
密码:ykfy

然后在项目根目录下新建一个名为pretrained的文件夹，把vgg16的模型文件放入pretrained文件夹中。

#### 3.2下载COCO数据集
[下载地址](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)
把解压后的train2014文件夹放在项目根目录下。

#### 3.3创建新的yml文件
找一个新的风格的图片，比如找一个火的图片，路径在img/fire.jpg。复制conf文件夹中wave.yml文件，然后改名fire.yml。把fire.yml中的：
style_image: img/wave.jpg
naming: "wave"
改为
style_image: img/fire.jpg
naming: "fire"

#### 3.4训练新的图像风格
```
python train.py -c conf/fire.yml

```


## 4.视频的风格转换
需要安装opencv，安装方式：
pip install opencv-python

准备好一个视频文件，然后在项目根目录下执行
python video.py --model_file models/wave.ckpt-done --video_file video/a.mp4

--model_file 是模型的路径，可以选择7个模型中的一个
--video_file是视频文件的路径

**视频**：
[视频效果](https://v.qq.com/x/page/w0812x561pe.html)


喜欢的朋友，记得star和follow哦。

