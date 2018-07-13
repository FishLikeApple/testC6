# -- coding: utf-8 --
'''
此文件实现迁移学习的主流程。
'''
import os.path
import tensorflow as tf
import JPG_to_npy
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import numpy as np
import gc

#image_path = "flower_photos"    
#JPG_to_npy.JPG_to_npy(image_path)  #文件转换

ckpt_file = "D:/Backup/Documents/Visual Studio 2015/inception_v3.ckpt"
output_file = "D:/Backup/Documents/Visual Studio 2015/Projects/testC6/testC6/trained_model"  #①
trainable_scopes = ["InceptionV3/Logits", "InceptionV3/AuxLogits"]

learning_rate = 0.0001
steps = 300
batch = 32
number_of_calsses = 5
training_file_pointer = 0

non_trainable_variables = []
images = tf.placeholder(tf.float32, [None, 299, 299, 3])
labels = tf.placeholder(tf.int64, [None])


#下面两个是分段加载的训练和评估函数。这里没有给出路径参数，并且用的默认是默认路径。
def segmented_batch_training(sess, training_step, batch, training_file_pointer):
    #gc.collect()  
    training_images = []                                
    training_labels = []
    for  k in range(batch):
        try:
            processed_data = np.load("data/training_data_"+str(training_file_pointer)+".npy")
        except:
            training_file_pointer = 0
            processed_data = np.load("data/training_data_"+str(training_file_pointer)+".npy")
        training_images.append(processed_data[0])
        training_labels.append(processed_data[1])
        training_file_pointer += 1
    _, loss = sess.run([training_step, tf.losses.get_total_loss()],
                       feed_dict={images: training_images, labels: training_labels})
    print(loss)

#下评估函数默认每次加载50个文件以进行评估，累积结果，并且最终输出平均结果
def segmented_evaluation(sess, evaluation_step, is_validation=True, segment_size=50): 
    #gc.collect()
    file_pointer = 0
    accumulated_accuracy = 0

    if is_validation == True: 
        front_file_name = "data/validation_data_"
    else:
        front_file_name = "data/test_data_"

    while True:
        evaluation_images = []                                
        evaluation_labels = []
        for i in range(segment_size):
            try:
                processed_data = np.load(front_file_name+str(file_pointer)+".npy")
            except:
                if (file_pointer%segment_size) != 0:
                    evaluation_images.append(processed_data[0])
                    evaluation_labels.append(processed_data[1])
                    accuracy = sess.run(evaluation_step, 
                                        feed_dict={images: evaluation_images, labels: evaluation_labels})
                    accumulated_accuracy += accuracy * (file_pointer%segment_size)
                return accumulated_accuracy / file_pointer
            evaluation_images.append(processed_data[0])
            evaluation_labels.append(processed_data[1])
            file_pointer += 1
        accuracy = sess.run(evaluation_step,
                            feed_dict={images: evaluation_images, labels: evaluation_labels})
        accumulated_accuracy += accuracy * segment_size


def training_start():  #主要函数

    #下面创建inception_v3模型的结构和其所有变量。
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, number_of_calsses)
        
    for var in tf.trainable_variables():  #区分可训练变量
            if var.op.name.startswith(trainable_scopes[0]) or var.op.name.startswith(trainable_scopes[1]): 
                tf.add_to_collection("trainable_variables_for_now", var)
            else:
                non_trainable_variables.append(var)

    tf.GraphKeys.TRAINABLE_VARIABLES = "trainable_variables_for_now"  #改变之前的可训练变量集合  ②

    load_fn=slim.assign_from_checkpoint_fn(ckpt_file, non_trainable_variables, True)  #读取不可训练变量

    tf.losses.softmax_cross_entropy(tf.one_hot(labels, number_of_calsses), logits)  #添加损失。原模型里已经有正则项。
    training_step = tf.train.RMSPropOptimizer(learning_rate).minimize(tf.losses.get_total_loss())  #定义优化训练步骤
    #print(tf.get_collection("losses"))  #③
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    load_fn(sess)

    for i in range(steps):
        segmented_batch_training(sess, training_step, batch, training_file_pointer)
        if (i+1) % 30 == 0:
            #saver.save(sess, output_file, global_step=i)  #保存checkpoint模型结构与数据
            print("%d-%dth iteration is passed with validation accuracy of %f" 
                  % (i-28, i+1, segmented_evaluation(sess, evaluation_step, True)))

    print("Final test accuracy is %f" % (segmented_evaluation(sess, evaluation_step, False)))

training_start()
'''
1，①处不能用局部路径？
2，我没在书上看到定义可训练变量的代码。我用的是②处的方法。
3，③处只会给出交叉熵损失，并没有正则项？
'''
