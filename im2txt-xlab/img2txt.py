
# coding: utf-8

# In[1]:

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

# In[2]:


# 训练好的模型存放路径
checkpoint_path = "./model/model.ckpt-3000000"
# 词汇表
vocab_file = "./im2txt/data/word_counts.txt"
# 图片路径
input_files = "./images/"


# In[8]:


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
         
    # 循环文件夹
    for root,dirs,files in os.walk(input_files):
        for file in files:
            # 打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            # 载入图片
            image = tf.gfile.FastGFile(os.path.join(root,file), 'rb').read()   
            # 获得图片描述
            captions = generator.beam_search(sess, image)
            # 打印多个标题
            for i, caption in enumerate(captions):
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if i == 0:
                    title = sentence
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


