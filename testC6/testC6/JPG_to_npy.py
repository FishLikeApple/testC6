# -- coding: utf-8 --
'''
此文件实现JPG到npy的转换，其中对每个JPG源文件都会生成一个npy目标文件。
文件名“test_data_28.npy”表示测试数据的第29个文件（第一个文件的文件号为0）。
'''
import glob
import os.path
import tensorflow as tf
import numpy as np
import gc
from tensorflow.python.platform import gfile

#下函数会由JPG文件得到npy文件
def JPG_to_npy(input_JPG_path, output_file_path = "data", validation_data_ratio = 0.1, 
               test_data_ratio = 0.1):
    file_list = []
    file_labels = []

    #获取所有文件和其标签
    sub_dirs = [x[0] for x in os.walk(input_JPG_path)]  #获取input_JPG_path下的所有子目录名
    extensions = ["jpg", "jpeg", "JPG", "JPEG"]
    current_label = 0
    for sub_dir in sub_dirs:
        if sub_dir == input_JPG_path: continue
        for extension in extensions:
            file_glob = glob.glob(sub_dir+"/*."+extension)
            file_list.extend(file_glob)   #添加文件路径到file_list
            file_labels.extend(np.ones(np.shape(file_glob))*current_label)   #添加标签到file_labels，标签与文件路径数量相同
        current_label +=1

    #打乱文件和标签
    state = np.random.get_state()
    np.random.shuffle(file_list)
    np.random.set_state(state)
    np.random.shuffle(file_labels)

    traning_count = 0
    test_count = 0
    validation_count = 0
    iteration_times = 0
    sess = tf.Session()   #获取图片数据时会用到
    for file_name in file_list:
        print("label=" + str(file_labels[iteration_times]) + "  file_path=" + file_name)   #打印当前储存的文件和标签
        image = tf.image.decode_jpeg(gfile.FastGFile(file_name, "rb").read())
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, [299, 299])   #①
        image_value = sess.run(image)

        chance = np.random.random_sample()#②
        if chance < validation_data_ratio:
            np.save(output_file_path+"/validation_data_"+str(validation_count)+".npy", 
                    np.asarray([image_value, file_labels[iteration_times]]))
            validation_count += 1
        elif chance < (validation_data_ratio + test_data_ratio):
            np.save(output_file_path+"/test_data_"+str(test_count)+".npy", 
                    np.asarray([image_value, file_labels[iteration_times]]))
            test_count += 1
        else:
            np.save(output_file_path+"/training_data_"+str(traning_count)+".npy", 
                    np.asarray([image_value, file_labels[iteration_times]]))
            traning_count += 1

        iteration_times += 1
        gc.collect()
'''
1，书上给的程序是把所有的图片数据都暂时存在了内存中。我的笔记本只有4g内存，显然这种
方式对我不合适。所以我就把其改写成：每个JPG源文件都会生成一个对应的npy目标文件，并
且生成完之后回收内存。
2，①为什么在会有不一样的格式？（我把这段去掉后确实loss有些不同）
3，文件列表已经被随机化了，那么②处的随机过程有什么意义？
'''