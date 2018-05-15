# coding: utf-8 
import tool
import tensorflow as tf
from tool import model_fn
from tool import input_fn
from tool import metric_fn
from ConsistentHashing import loadHashList
from ConsistentHashing import allocateData2
from ConsistentHashing import static_ip_load
from ConsistentHashing import flatIpLoad
def main():
    total_size = 50#子模型总数据量
    hash_size = 100#子model harsh 表的大小
    batch_size = 100#监督学习下推荐使用 非监督batch_size 和total size 应该是一样的

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(session_config=session_config)

    params = {
        'layer_sizes': [128, 256],
        'net_name': 'first_layer',
        'total_size': total_size,
        'hash_size': hash_size,
        'batch_size': batch_size,
        'learning_rate': 0.001 ,
        'log_every_n_iter': 100,
        'model_path': 'model/digit/supervised_model',
        'log_path': 'result_log/digit100_125/',
        'log_file': '{}supervised_log.txt'.format('' ),
        'max_steps': 100000 ,
        'is_retrain': True
    }
    estimator = tf.estimator.Estimator(model_fn, params['model_path'], config=run_config, params=params)
    train_spec = tf.estimator.TrainSpec(
        lambda: input_fn(1, total_size, 'linear', 'uniform', True, batch_size),
        max_steps=params['max_steps'])
    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(1, total_size, 'linear', 'uniform', False), 1, 'eval')
    # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    with tf.Session() as sess:
        sess.run( input_fn(1, total_size, 'linear', 'uniform', False))
    res=estimator.predict(lambda: input_fn(1, total_size, 'linear', 'uniform', False))
    #print([r for r in res])
    #for r in res:
    #    print(r)
    return res
    #with tf.Session()as sess:
    #    sess.run(res)
        #predict api
def generate_learnedHashList(res):
    hashList = loadHashList()
    learnedHashList = []
    for i in hashList:
        i[1]=[]
    for i in res:
        hashList[i["pos"][0]][1].append(i["inputs"][0])
    return hashList

res = main()
learnedHashList = generate_learnedHashList(res)
#print(learnedHashList)
ipload = allocateData2(learnedHashList)
print(ipload)
ipload = flatIpLoad(ipload)
print(ipload)
#print(ipload)
static_ip_load(ipload)
#print(learnedHashList)
#print(ipload)
#print(ipload)


