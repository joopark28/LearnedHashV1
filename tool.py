# coding: utf-8 
import tensorflow as tf
import numpy as np
from data import generate_digit_data
from LogHook import LogHook
from data import generate_digit_data

class supervisedModel:
    """
    layer_sizes ：这个神经网络每一层的神经元的数量
    return_size :想当与output 有多少个神经元 ，监督学习return size 是1 非监督是harsh 表的大小
    """
    def __init__(self, layer_sizes, return_size):
        self.num_layers = len(layer_sizes)
        assert self.num_layers < 4
        self.layer_sizes = layer_sizes
        for layer_size in layer_sizes:
            assert layer_size > 0
        self.return_size = return_size

    def __call__(self, inputs):
        y = inputs
        # print(inputs.shape)
        for layer_size in self.layer_sizes:
            y = tf.layers.batch_normalization(tf.layers.dense(y, layer_size, activation=tf.nn.relu))
            # 对每一层建立一个全连接层
        y = tf.layers.dense(y, self.return_size)
        return y
    
def model_fn(features, labels, mode, params):
    """model_fn constructs the ML model used as a hash function."""
    """
features 是key  labels 是output的position 
mode 有3种模式 是训练预测还是计算,训练
任意其他的 参数可以通过 params 传入
    """
    inputs = features
    if isinstance(inputs, dict):
        inputs = features['inputs']

    layer_sizes = params['layer_sizes']
    net_name = params['net_name']
    total_size = params['total_size']
    hash_size = params['hash_size']
    batch_size = params['batch_size']
    # 向学长 自己写的 表示训练多少次去统计一次数据
    log_every_n_iter = params['log_every_n_iter']

    """
    如果是监督学习 根据 上面的model 建立模型
    """
    # 监督学习
    model = supervisedModel(layer_sizes, 1)
    logits = model(inputs)  # 正向传播结果


    # 预测
    """
    如果是监督学习 logits就是 location 对他取证
    if 是非监督学习 取 logits最大的 为他的位置
    """
    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions={'inputs':inputs, 'pos':tf.to_int32(tf.round(logits))})


    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(tf.to_int32(labels), tf.to_int32(tf.round(logits)))))  # 结果和label的差距
    # loss = tf.nn.l2_loss((labels - logits))
    loss = tf.losses.mean_squared_error(labels, logits)  # loss使用logits和labele的均方误差

    # 统计每个slot的数据量有多少key 只用到了counts
    values, _, counts = tf.unique_with_counts(tf.reshape(tf.mod(tf.to_int32(tf.round(logits)), hash_size), [-1]))
    hashed_logits = tf.expand_dims(tf.to_float(tf.sparse_to_dense(values, [hash_size], counts, 0, False)), 1)
    # 统计方差，仅在N条数据分配到M个slot中使用方差评估，N>>M
    # 1. var = tf.reduce_mean(tf.pow(tf.subtract(tf.to_float(counts), total_size / hash_size), 2))
    var = tf.constant(-1)
    # var means 方差 ， 在10000映射到100时 1起作用

    # 训练 model 第一次一定是进这个 if 里面
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 平均查找次数和空余空间
        avg_search = tf.reduce_sum(counts * (counts + 1) / 2) / batch_size
        # 计算它的平均查找次数
        spare_space = 1 - tf.divide(tf.count_nonzero(counts), hash_size)
        # 计算他空余的slot 的数量
        global_step = tf.train.get_global_step()
        # 本model 当前训练迭代的次数

        # 添加训练时的hook
        train_hooks = [LogHook(params['log_path'], params['log_file'], log_every_n_iter,
                               {'loss': loss, 'avg': avg_search, 'accuracy': accuracy, 'spare': spare_space,
                                'var': var}, {'avg': 'min', 'spare': 'min', 'var': 'min'}, total_size, hash_size)]

        learning_rate = params['learning_rate']
        optimizer = tf.train.AdamOptimizer(learning_rate)
        """
        真正的训练op
        """
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=optimizer.minimize(loss, global_step),
                                          training_chief_hooks=train_hooks)
    # 在训练的过程中 评估模型的好坏
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=metric_fn(labels, logits, counts,
                                                                    hash_size, total_size))

def input_fn(index, total_size, method, distribute, training, batch_size=None):
    keys_list = generate_digit_data('load', distribute)[index][:total_size]
    # keys_list = [i+0.01 for i in range(1000)]

    keys_list, labels = cal_labels(keys_list, method)
    #print(np.shape(keys_list))
    #print(labels)
    dataset = tf.data.Dataset.from_tensor_slices(({'inputs': keys_list}, labels))
    if training:
        assert batch_size != None
 #       dataset = dataset.shuffle(10 * total_size).repeat().batch(batch_size)
        dataset = dataset.shuffle(10 * total_size).batch(batch_size)

    else:
        dataset = dataset.batch(total_size)
        # evaluation  和traiing  进入这个口，
    return dataset.make_one_shot_iterator().get_next()

def cal_labels(keys=None, method='linear', hash_num=26431):
    assert method == 'linear' or method == 'multiply' or method == 'None'
    if method == 'linear':
        keys = np.sort(keys)
    keys = np.expand_dims(keys, 1)
    if method == 'multiply':
        labels = np.round(keys * hash_num)
    else:
        labels = np.expand_dims(np.arange(len(keys), dtype=np.float64), 1)
    return keys, labels

def metric_fn(labels, logits, counts, hash_size, total_size):
    avg_search = tf.metrics.mean(tf.reduce_sum(counts * (counts + 1) / 2) / total_size, name='avg_search')
    spare_space = tf.metrics.mean(1 - tf.divide(counts, hash_size), name='spare_space')
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.to_int32(tf.round(logits)), name='accuracy')
    return {"accuracy": accuracy, "avg_search": avg_search, "spare_space": spare_space}
a,b = input_fn(1, 50, 'linear', 'uniform', False)
keys_list = generate_digit_data('load', 'uniform')[1][:50]
