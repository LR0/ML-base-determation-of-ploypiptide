# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:24:14 2019

@author: Administrator
"""
import tensorflow as tf
from sklearn import datasets
#import matplotlib.pyplot as plt

#定义训练数据batch的大小
batch_size = 8
#产生128组数据
dataset_size = 128
#设置训练的轮数
STEPS = 5000

#定义test数据集与train训练集的数据准确率
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={x:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    #tf.cast：用于改变某个张量的数据类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
    result = sess.run(accuracy,feed_dict={x:v_xs,y:v_ys})
    return result


def add_layer(inputs,in_size,out_size,activation_function=None): #输入参数inputs，行列矩阵维数in_size等，激励函数
     #定义权重为行(in_zise)列(out_size)的随机矩阵
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) 
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)           #定于误差
    Wx_plus_b = tf.matmul(inputs,Weights) + biases               #线性运算，W*x+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs



x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='samples')#特征为表现为两个数，所以shape为2
y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='labels')#数据归结为两类，所以shape为2

#add output layer
#一般在分类学习用，使用tf.nn.softmax
hide1 = add_layer(x,2,32,activation_function=tf.tanh)
hide2 = add_layer(hide1,32,8,activation_function=tf.tanh)
prediction = add_layer(hide2,8,2,activation_function=tf.nn.softmax)

#the error between prediction and real data
#loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),reduction_indices=[1]))  #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#学习率为0.5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #生成聚类图像
    X,temp=datasets.make_moons(128,noise=0.2)
    Y=[]
    for i in temp:
        if i==0:
            Y.append([0,1])
        else:
            Y.append([1,0])
#    #打印聚类图像
#    mark = ['or', 'ob']
#    #这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
#    color = 0
#    j = 0 
#    for i in Y:
#        plt.plot([X[j:j+1,0]], [X[j:j+1,1]], mark[i[0]], markersize = 5)
#        j +=1
#    plt.show()
    
    for i in range(STEPS+1):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
	    #每次选取batch_size个样本进行训练
    
        sess.run(train_step,feed_dict={x:X[start:end],y:Y[start:end]})
	    #通过选取的样本训练神经网络并更新参数
    
        if i%1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y:Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i,total_cross_entropy))
