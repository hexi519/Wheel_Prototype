import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import scale,MinMaxScaler
import torch.utils.data
scaler = MinMaxScaler()

import logging
logger = logging.getLogger("hesy.utils")

class TrafficData(torch.utils.data.TensorDataset):
    '''数据的装饰类'''
    def __init__(self, input_data,time_stamp,test_index,train ):
        '''
        将输入的数据转换成如下shape
        (time_index, day_index, city_index) * batch_size * seq_len

        Args
            input_data
                size : city day time （ 66 xx 144 ）
            test_index ( tuple )
                2*list(66)  66个城市每个城市测试集包含了两个测试样例
                这里存储的是日子的下标 , 依次是 (weekend,workday),
            train ( bool )
                是否装载的是训练数据，会影响数据的格式
    
        Returns 
         
        '''
        self.time_stamp = time_stamp
        self.train = train
        self.data = torch.tensor( input_data,dtype=torch.float).reshape(-1,1)
        self.label = None

        city_input = scaler.fit_transform(np.array(range(66)).reshape(-1,1))+0.1
        day_input = scaler.fit_transform(np.array(range(21)).reshape(-1,1))+0.1
        time_input = scaler.fit_transform(np.array(range(144)).reshape(-1,1))+0.1
        
        if train:
            self.label = torch.tensor([  
            [time_input[time_index],day_input[day_index],city_input[city_index] ] for city_index in range(66)\
                            for day_index in range(21) if day_index not in [test_index[0][city_index],test_index[1][city_index]] \
                            for time_index in range(144) ],dtype = torch.float).reshape(-1,3)

        else:
            self.label = torch.tensor([  
            [time_input[time_index],day_input[day_index],city_input[city_index]] for city_index in range(66)\
                            for day_index in [ test_index[0][city_index],test_index[1][city_index] ] \
                            for time_index in range(144) ],dtype = torch.float).reshape(-1,3)
            
        # debug_logger.info(f"self.label shape is \n\t{self.label.shape}")
        # debug_logger.info(f"self.label is {self.label}")

    
    def __getitem__(self, index):
            # city * day * time_index
        # return self.label[index*self.time_stamp%144 : (index+1)*self.time_stamp%144 if (index+1)*self.time_stamp%144 else 144],\
        #         self.data[index*self.time_stamp%144 : (index+1)*self.time_stamp%144 if (index+1)*self.time_stamp%144 else 144]
        return self.label[index*self.time_stamp : (index+1)*self.time_stamp],\
                self.data[index*self.time_stamp : (index+1)*self.time_stamp]
    
    def __test__(self):
        pass

    def __len__(self):
        return 144*19*66//self.time_stamp if self.train else 144*2*66//self.time_stamp

class RegressionRnn(nn.Module):
    def __init__(self,input_dim ,out_dim ,num_hidden, num_layer,drop_rate=0, **kwargs):
        super(RegressionRnn, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size=input_dim , hidden_size = num_hidden, num_layers = num_layer, dropout=drop_rate,batch_first =True)
        self.out = nn.Linear(num_hidden, out_dim)
        self.flag=True

    def forward(self, inputs, state):
        '''
            args 
            
            returns
                input size is [8, 12, 3]  &  output size is [8, 12, 1]
        '''
#         output, _ = self.rnn(inputs,torch.zeros(2, 8, 128))
#         import ipdb ; ipdb.set_trace()
        output, _ = self.rnn(inputs,state ) 
        output = self.out(output)
        if self.flag==True:
            # logger.critical(f"In RegressionRnn : output shape is {output.shape}")
            # logger.critical(f"In RegressionRnn : output is {output}")
            self.flag=False
#         output.squeeze_()  # ?? 为什么会有这个...
        return (output,state) #


class ParaList():
    """ 存储参数 和 数据存储文件 相关信息的类 
    
        通过 模型参数 和 当前时间 对本次的运行 给出一个 unique id 对应的存储文件目录( 就是在unique id前面加上logFile/ 后续的相关数据都放在该目录下面 )

        模型会有默认的参数列表 为了简化id 我们不使用全部的模型参数 只使用修改了的模型参数
        
        unique id
            e.g. time_stamp-12_num_epochs-200_  [ 如果参数都没有改变 默认就是_文件 ]
    
        __init__
            Args
                para ( dict )
                    从configs.config加载进来的默认参数列表
                    e.g. { "time_stamp":12 , "num_epochs":200 }
        __call__     
            直接使用ParaList的实例并传入属性字符串 则会返回其相应的属性值
            
            使用说明
                >>> ParaList para
                >>> para("num_epochs")
                200
    """
    # TODO 或许可以考虑使用property装饰器来做 https://blog.csdn.net/sj349781478/article/details/79546666  这个tldr  好好学习下
    def __call__(self,str):

        return self.para[str]
    def __init__(self,para):
        self.para = para
        self.orgin_filename = self.__time__()
        self.curr_filename = self.orgin_filename
        self.changed_para = {}
        
    def GetSaveDir(self):
        """根据当前模型跑的参数和时间 在logDir文件夹下面创建记录```文件夹``` """
        return "logFile/"+self.curr_filename  
        # return self.curr_filename  
    
    def __time__(self):
        """保存的文件的结尾使用时间标识
        
            Returns
                时间字符串， e.g. "_1201-1103" 表示程序是12月1号11点03分开始进行的

            .. warning:: 不要使用time作为当前的文件命名，本函数实际上只返回 _ , 表示初始时刻没有参数被修改，使用默认的参数列表   
        """
        # import time
        # save_name = time.strftime("_%m%d-%H%M", time.localtime())
        save_name = "_"
        return save_name
    
    def ChangePara(self,para_dict):
        """修改数据

            Args
                para_dict (dict)
                    通过命令行传入的 在默认参数列表上进行修改的参数列表
                    
                    e.g. { "time_stamp":"12" , "num_epochs":"200" }
        """
        import re 
        after_save = self.orgin_filename
        for key,value in para_dict.items():
            characters = [f"_{para_name}-{value}" for para_name ,value in para_dict.items() ]
            for ch in characters:
                after_save+=ch
            
            self.para[key] = float(value) if("." in value ) else int(value)
        
        logger.debug(f"self.para is {self.para}")
        self.curr_filename = after_save