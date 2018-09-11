import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# 通过操作符号变量来描述可交互的操作单元
x = tf.placeholder("float",[None, 784])

# x不是一个特定的值，而是一个占位符placeholder
# 我们在TensorFlow运行计算时输入这个值
# 我们希望能够输入任意数量的MNIST图像，每个图展平成784维的向量。
# 我们用2维的浮点数张量来表示这些图，这个张量的形状是[None, 784]。（这里的None表示此张量的第一个维度可以是任何长度的）



# 模型需要权重值和偏置值，当然可以把它们当作另外的输入（使用占位符）
# 但Tensorflow有一个更好的方法来表示：Variable
# 一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 实现模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 为了计算交叉熵，需要首先添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None,10])

# 首先用tf.log()计算y的每个元素的对数。
# 接下来，我们把y_的每一个元素与其对应相乘。
# 最后用tf.reduce_sum计算张量的所有元素的总和
# 这里交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵总和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# TensorFlow拥有一张描述你各个计算单元的图，可以自动地使用反向传播算法
# 来有效确定变量如何影响cost值，并根据你选择的优化算法不断修改变量以降低成本

# 这里要求TensorFlow用梯度下降算法以0.01的学习速率最小化交叉熵
# TensorFlow在这里实际上所做的是，在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 下面循环的每个步骤中，随机抓取训练数据中的100个批处理数据点
# 然后用这些数据点作为参数替换之前的占位符来运行train_step
# 随机梯度训练既可以减少计算开销，又可以最大化地学习数据集的总体特征。

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
# 评估我们的模型
# 首先找出那些预测正确的标签。
# tf.argmax()是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0，1组成，因此最大值1所在的索引位置就是类别标签
# 比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而tf.argmax(y_,1)代表正确的标签
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 上行代码会给我们一组布尔值。我们可以将布尔值转换为浮点数，然后取平均数得到准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
