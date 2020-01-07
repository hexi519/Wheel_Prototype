"""存储模型的默认参数"""
paras_dict={
    'time_stamp': 12,
    'batch_size':8,
    'num_workers': 0,
    'num_epochs':200,
    'lr' : 0.01,
    'clip' : 1, 

    'num_hidden' : 128,
    'num_layer' : 2,
    'drop_rate' : 0.5,

    'window_cover' : False,
    'diff_data' : False,
    'device':"cuda:1"
}
