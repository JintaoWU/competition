import tensorflow as tf
import numpy as np
from numpy import loadtxt
import math
import matplotlib.pyplot as plt
#精度计算
def Accuracy(y_pred,y_test):
    mae = 1 - abs((y_pred - y_test) / y_test)
    return np.mean(mae)

#数据归一化
def normal(X):
    for i in range(len(X[0])):
        tmp=list(X[:,i:i+1])
        maxV=max(tmp)
        minV=min(tmp)

        for j in range(len(tmp)):
            X[j][i]=(X[j][i]-minV)/(maxV-minV)

def getBatch(X,Y,i):
    num=20
    index=i*num%60
    return X[index:index+num,:],Y[index:index+num,:]

def rmse(y_pred,y_test):
    sum=0.0
    for i in range(len(y_pred)):
        sum=sum+(y_pred[i]-y_test[i])*(y_pred[i]-y_test[i])
    return math.sqrt(sum/len(y_test))
def nrmse(y_pred,y_test):
    e=rmse(y_pred,y_test)
    m=np.mean(y_test)
    return e/m

#画图
def plot(x,y):
    plt.scatter(x, y)
    # 设置title和x，y轴的label
    #plt.title("Height And Weight")
    plt.xlabel("real")
    plt.ylabel("pred")
    # 展示图片 *必加
    plt.show()
    plt.close()



X_data = loadtxt('tmp', dtype=float, delimiter="\t")
Xtmp=X_data[:, 0:3]
#normal(Xtmp)


X_train = Xtmp[0:60, 0:2]
y_train = X_data[0:60, 2:3]

X_test = Xtmp[0:80, 0:2]
y_test = X_data[0:80, 2:3]



# 2.Create the model
# y=wx+b
x = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 1])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.


# 损失函数
loss = tf.reduce_sum(tf.square(y_ - y))

# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
train_step = tf.train.GradientDescentOptimizer(0.0000000001).minimize(loss)

#观测值的均值
y_mean=tf.reduce_mean(y_)
R2FZ=tf.reduce_sum(tf.square(y - y_mean))
R2FM=tf.reduce_sum(tf.square(y_ - y_mean))
R2=tf.div(R2FZ, R2FM)

SST=tf.reduce_sum(tf.square(y_ - y_mean))
SSR=tf.reduce_sum(tf.square(y - y_mean))
SSE=tf.reduce_sum(tf.square(y - y_))

r21=tf.div(SSR, SST)
r22=tf.div(SSE, SST)

error = tf.abs(tf.subtract(y, y_))
Probability = tf.subtract(1.0, tf.div(error, y_))
accuracy = tf.reduce_mean(tf.cast(Probability, tf.float64))
MAE = tf.reduce_mean(error)

RMSE=tf.sqrt(tf.reduce_mean(tf.square(y_-y)))

# Init model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for i in range(10001):
    batch_xs,batch_ys=getBatch(X_train,y_train,i)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #print(loss.eval(feed_dict={x: batch_xs, y_: batch_ys}))
    if (i % 100 == 0):
        y_MAE=MAE.eval(feed_dict={x: X_test, y_: y_test})
        y_RMSE=RMSE.eval(feed_dict={x: X_test, y_: y_test})
        y_R2=R2.eval(feed_dict={x: X_test, y_: y_test})
        y_Accu=accuracy.eval(feed_dict={x: X_test, y_: y_test})
        y_r21=r21.eval(feed_dict={x: X_test, y_: y_test})
        y_r22=1.0-r22.eval(feed_dict={x: X_test, y_: y_test})
        y_nrmse=y_RMSE/y_mean.eval(feed_dict={y_: y_test})

        ypre_MAE = MAE.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_RMSE = RMSE.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_R2 = R2.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_Accu = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_r21 = r21.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_r22 = 1.0 - r22.eval(feed_dict={x: batch_xs, y_: batch_ys})
        ypre_nrmse = y_RMSE / y_mean.eval(feed_dict={y_: batch_ys})
        print("iter:%d-th"%i)
        print("pred:\t\tMAE=%f\t\tRMSE=%f\t\tR2=%f\t\tAccu=%f\t\ty_r21=%f\t\ty_r22=%f\t\tnRMSE=%f\n"%(y_MAE,y_RMSE,y_R2,y_Accu,y_r21,y_r22,y_nrmse))
        print("real:\t\tMAE=%f\t\tRMSE=%f\t\tR2=%f\t\tAccu=%f\t\ty_r21=%f\t\ty_r22=%f\t\tnRMSE=%f\n" % (ypre_MAE, ypre_RMSE, ypre_R2, ypre_Accu,ypre_r21,ypre_r22,ypre_nrmse))
        print("------------------------------------------------------------------------------")
#tf.train.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')

w=sess.run(W)
p=np.zeros(len(X_train[0]))
for i in range(len(p)):
    p[i]=w[i][0]
print("参数w：%s"%p)
b=sess.run(b)[0]
print("参数b:%f"%b)

def predV(x_test,y_test):
    v=np.zeros(len(y_test))
    for i in range(len(y_test)):
        sum_=0.0
        for j in range(len(p)):
            sum_=sum_+x_test[i][j]*p[j]
        sum_=sum_+b
        v[i]=sum_
    return v

pred=predV(X_test,y_test)
print("-----pred----")
for i in range(len(pred)):
    print(str(pred[i])+"\t")
print("\n")

real=np.zeros(len(y_test))
for i in range(len(real)):
    real[i]=y_test[i][0]
print("-----real----")
for i in range(len(real)):
    print(str(real[i])+"\t")
print("\n")

plot(real,pred)
#print("W1:", sess.run(W))  # 打印v1、v2的值一会读取之后对比
#print("W2:", sess.run(b))

