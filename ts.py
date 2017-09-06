import tensorflow as tf

x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'yy')
f = x*y

c = tf.constant(1)

print(x.graph)

with tf.Session() as sess:
    print(c.eval())
    init = tf.global_variables_initializer()
    init.run()
    #x.initializer.run()
    #y.initializer.run()
    result = f.eval()
    #print(result)
    #print(x.eval())