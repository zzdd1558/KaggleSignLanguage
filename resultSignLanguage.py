from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import random
import datetime

x_data = np.load('./SignLanguageDataset/X.npy')
y_data = np.load('./SignLanguageDataset/Y.npy')

# 설정 값
'''
image_size :  이미지 사이즈는 64 x 64
image_size_flat : reshape하여 64 * 64의 배열을 4096개의 하나의 배열로 변환하기 위한 값
num_classes : 0 ~ 9까지이기 때문에 10개
training_iters : 트레이닝시킬 데이터의 갯수
batch_size : 배치 사이즈
display_step : print로 찍어주기위한 값 ( batch_size * display_step 배수일때만 출력 )
dropout :
learning_rate :
 - learning_rate 값 잘줘야합니다. 0.1로 했을경우 acc : 10% 간당간당.

test_size : train 데이터와 test 데이터로 나눈다. ex) 0.15  - train_data 85% , test_data 15%가 된다.
'''

image_size = 64
image_size_flat = image_size * image_size
num_class = 10
training_iters = 40000
batch_size = 16
display_step = 100
dropout = 0.75
learning_rate = 0.001
test_size = 0.15

'''
크로스 벨리데이션 사용을 위해 사이킷런의  train_test_split 사용
train-test_spilt =>   return 트레이닝데이터 , 테스트데이터 , 트레이닝레이블 , 테스트 레이
'''
X_train, X_test, Y_train, Y_test = \
    train_test_split(x_data, y_data, test_size=test_size, random_state=42)

train_X = X_train
train_Y = Y_train

new_train_X = train_X.reshape(X_train.shape[0], image_size_flat)
new_test_X = X_test.reshape(X_test.shape[0], image_size_flat)


with tf.name_scope('input_placeholder') as scope :
    x = tf.placeholder(tf.float32, shape=[None, image_size_flat], name='x_data')
    y_ = tf.placeholder(tf.float32, shape=[None, num_class], name='y_data')
    keep_prob = tf.placeholder(tf.float32)

    # output shape
    #print('Shape of placeholder', x.shape, y_.shape)

def conv2d(x, W, b):
    # x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    # x = tf.nn.bias_add(x, b)

    x = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME"), b)

    return tf.nn.relu(x)


def max_pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def conv_layer(x, weights, bias, dropout):
    x = tf.reshape(x, [-1, 64, 64, 1])

    with tf.name_scope('convolution_and_maxPooling_1') as scope:
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = max_pool(conv1)

    with tf.name_scope('convolution_and_maxPooling_2') as scope:
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = max_pool(conv2)

    with tf.name_scope('fully_Connected') as scope:
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('drop_out') as scope:
        fc1 = tf.nn.dropout(fc1, dropout)
        y_conv = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return y_conv


def next_batch(X, Y, batchSize=batch_size):
    dataLength = X.shape[0]
    count = 0

    while count < dataLength / batch_size:
        # 랜덤값 변환을 위한 시드값 변경 ( 현재 시간으로 시드 변경 )
        random.seed(datetime.datetime.now())
        randomIndex = random.randint(0, dataLength - batch_size - 1)
        count += 1
        yield (X[randomIndex:randomIndex + batchSize], Y[randomIndex:randomIndex + batchSize])


def setVariableRandom(shape, name):
    return tf.Variable(tf.random_normal(shape), name=name)


weights = {
    'wc1': setVariableRandom([5, 5, 1, 32], 'wc1'),
    'wc2': setVariableRandom([5, 5, 32, 64], 'wc2'),
    'wd1': setVariableRandom([64 * 64 * 4, 1024], 'wd1'),
    'out': setVariableRandom([1024, num_class], 'wout')
}

biases = {
    'bc1': setVariableRandom([32], 'bc1'),
    'bc2': setVariableRandom([64], 'bc2'),
    'bd1': setVariableRandom([1024], 'bd1'),
    'out': setVariableRandom([num_class], 'bout')
}

with tf.name_scope('model') as scope:
    model = conv_layer(x, weights, biases, keep_prob)

with tf.name_scope('optimizer') as scope:
    cross_entoropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entoropy)

predict_step = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1

    #tensorBoard 사용하기 위한 설정. log_dir 디렉터리에 파일 생성.
    tw = tf.summary.FileWriter('log_dir', graph=sess.graph)

    while step * batch_size < training_iters:
        batch_result = next_batch(new_train_X, train_Y, batch_size)
        batch_x, batch_y = next(batch_result)

        sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            # _, loss = sess.run([train_step, cross_entoropy], feed_dict=fd)
            # acc = sess.run(accuracy_step, feed_dict=test_fd)
            print('*' * 15)
            loss, acc = sess.run([cross_entoropy, accuracy_step], feed_dict={x: batch_x,
                                                                             y_: batch_y,
                                                                             keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Loss= " + \
                  "{:.3f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        step += 1
    acc = sess.run(accuracy_step, feed_dict={x: new_test_X, y_: Y_test, keep_prob: 1.})
    print("정답률 = ", acc)
