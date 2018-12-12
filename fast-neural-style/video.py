
# coding: utf-8

# In[ ]:


# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import cv2
import sys

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("video_file", "a.mp4", "")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # Make sure 'generated' directory exists.
    generated_file = 'generated/res.mp4'
    if os.path.exists('generated') is False:
        os.makedirs('generated')
    if os.path.exists('video') is False:
        os.makedirs('generated')
    # Get image's height and width.
#     height = 0
#     width = 0
    # 从文件读取视频内容
    cap = cv2.VideoCapture(FLAGS.video_file)
    # 视频每秒传输帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 视频图像的宽度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频图像的长度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 视频帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(generated_file,fourcc,fps,(frame_width,frame_height))
#     with open(FLAGS.image_file, 'rb') as img:
#         with tf.Session().as_default() as sess:
#             if FLAGS.image_file.lower().endswith('png'):
#                 image = sess.run(tf.image.decode_png(img.read()))
#             else:
#                 image = sess.run(tf.image.decode_jpeg(img.read()))
#             height = image.shape[0]
#             width = image.shape[1]

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            
            # Read image data.
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            
            image = reader.get_image('video/temp.jpg', frame_height, frame_width, image_preprocessing_fn)

            # Add batch dimension
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            # Remove batch dimension
            generated = tf.squeeze(generated, [0])
            
            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)
            n = 0 
            while(True):
                # ret 读取成功True或失败False
                # frame读取到的图像的内容
                # 读取一帧数据
                ret,frame = cap.read()
                if ret!=True:
                    break
                n+=1
                cv2.imwrite('video/temp.jpg', frame)
            
                with open('video/temp2.jpg', 'wb') as img:
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                
                image = cv2.imread('video/temp2.jpg')
                out.write(image)
                sys.stdout.write('\r>> Converting image %d/%d' % (n, frame_count))
                sys.stdout.flush()

        cap.release()
        cv2.destroyAllWindows()
                


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


