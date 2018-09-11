import tensorflow as tf

# 创建一个常量op，产生一个1*2矩阵。这个op被作为一个节点
# 加到默认图中
#
# 构造器的返回值代表该常量op的返回值。
matrix1 = tf.constant([[3., 3.]])

# 创建另一个常量op，产生一个2*1矩阵
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op, 把'matrix1'和'matrix2'作为输入。
# 返回值'product'代表矩阵乘法的结果
product = tf.matmul(matrix1, matrix2)

'''
默认图现在有三个节点，两个constant() op,和一个matmul() op
为了真正进行矩阵相乘运算，并得到矩阵乘法的结果，你必须在会话里启动这个图

在一个会话中启动图：
构造阶段完成后，才能启动图。启动图的第一步是创建一个Session对象
如果无任何创建参数，会话构造器将启动默认图
'''

'''
# 启动默认图
sess = tf.Session()

# 调用sess的'run()'方法来执行矩阵乘法op，传入'product'作为该方法的参数
# 上面提到，'product'代表了矩阵乘法op的输入，传入它是向方法表明
# 我们希望取回矩阵乘法op的输出。
#
#整个执行过程是自动化的，会话负责传递op所需的全部输入。op通常是并发执行的
#
# 函数调用'run(product)'触发了图中三个op（两个常量op和一个矩阵乘法op）的执行
#
# 返回值'result'是一个numpy 'ndarray'对象
result = sess.run(product)
print(result)

# 任务完成，关闭会话
sess.close()
'''

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
