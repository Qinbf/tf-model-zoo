
# coding: utf-8

# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import tensorflow as tf
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import cv2


# In[3]:


# 训练好的模型存放路径
checkpoint_path = "./model/model.ckpt-3000000"
# 词汇表
vocab_file = "./im2txt/data/word_counts.txt"
# 视频路径
input_files = "./video/1.mp4"


# In[7]:


# 载入训练好的模型
g = tf.Graph()
with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)

# 载入词表
vocab = vocabulary.Vocabulary(vocab_file)

with tf.Session(graph=g) as sess:
    # 载入训练好的模型
    restore_fn(sess)
    generator = caption_generator.CaptionGenerator(model, vocab)
    
    # 从文件读取视频内容
    cap = cv2.VideoCapture(input_files)
    # 视频每秒传输帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 视频图像的宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频图像的长度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter('video/output.avi',fourcc,fps,(frame_width,frame_height))
    n = 0
    sentence = ''
    while(True):
        # ret 读取成功True或失败False
        # frame读取到的图像的内容
        # 读取一帧数据
        ret,frame = cap.read()
        if ret!=True:
            break
        n += 1
        # 每秒生成一个新的描述
        if n == fps:
            n = 0
            cv2.imwrite('video/temp.jpg', frame)
            # 载入图片
            image = tf.gfile.FastGFile('video/temp.jpg', 'rb').read()   
            # 获得图片描述
            captions = generator.beam_search(sess, image)
            # 获得图片描述
            sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
            sentence = " ".join(sentence)
            
        # cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
        frame = cv2.putText(frame, sentence, (50, frame_height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 写入frame
        out.write(frame)
    
cap.release()
cv2.destroyAllWindows()

