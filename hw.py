import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from sklearn.preprocessing import scale,MinMaxScaler
scaler = MinMaxScaler()

from configparser import ConfigParser
import time

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter  

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from configs import *
from tqdm import tqdm
# 后面最好改掉。。。污染了命名空间。。。
from utils import * 
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# ======  load configs and changed paras  to get save file name & 配置一系列文件========
import argparse
parser = argparse.ArgumentParser()
for key in paras_dict.keys():
    parser.add_argument( "--"+key )

parser.add_argument( "-hesy",help="这是个debug arg，用于测试"  )
parser.add_argument( "-f",help=" for case of jupyter" )
args = parser.parse_args()
# name  =[ f"_{key}-{value}" for (key,value) in vars(args).items() if value!= None and key!="dir" ]
changed_dict ={}
for (key,value) in vars(args).items():
  if value!= None and key!="dir" and key!="f" and key!="hesy":  # 删除掉无关的参数
    changed_dict[key]=value

para = ParaList(paras_dict)    # import from configs
para.ChangePara(changed_dict)

# =============================
import logging
LogTo( para.GetSaveDir())
logger = logging.getLogger("hesy.hw")
 
# =============================
# TODO 这里修改一下 统一把所有文件放在一个tensorboard的文件夹里面
writer = SummaryWriter( para.GetSaveDir()+"/tensorboard")
logger.info(f"save directory is { para.GetSaveDir()} ")

device = torch.device( para("device") if torch.cuda.is_available() else 'cpu')
logger.info(f"device is { device} ")

# ===========================处理数据集======================================
# from utils import TrafficData,RegressionRnn,ParaList
logger.info("loading data")

data  = sio.loadmat("./demand_10.mat")
traffic_data = data['demand'].reshape((-1,144,66)) # (21,144,66)
 
weekend_day = [ i-1 for i in [2,3,9,10,16,17] ]  #  日子的下标 （ from 0
weekend = [ weekend_day[random.randint(0,len(weekend_day))] for i in range(66) ] 
work_day = list(range(21))
[work_day.remove(i) for i in weekend_day]
workday = [ work_day[random.randint(0,len(work_day))] for i in range(66) ] 

# (city,day,time)
test_data = np.array([ traffic_data[[weekend[i],workday[i]],:,i] for i in range(len(weekend)) ])
train_data = np.array([ traffic_data[j,:,i] for i in range(66) for j in range(21) if j not in [weekend[i],workday[i]]  ])

# ======  load configs and changed paras  ========


def train(model, dataset, para, writer, device ,checkpoint=None):
  check_flag = True
  # TODO 
  if checkpoint !=None:
    logger.error("还没有实现这部分的逻辑...")

  train_dataset = TrafficData( 
            dataset ,
            time_stamp = para("time_stamp"),
            test_index=(weekend,workday),
            train=True
            )
  # TODO 这里要改成True
  data_iter = Data.DataLoader(train_dataset, para("batch_size"), shuffle=False,num_workers= para("num_workers") )

  optimizer = torch.optim.Adam(model.parameters(), lr= para("lr"))

  # loss = nn.MSELoss(reduction="none")  # ?? TODO 
  loss = nn.MSELoss()  # ?? TODO 
  
  min_loss=99999999
  
  # for epoch in tqdm(range( para("num_epochs"))):
  for epoch in range( para("num_epochs")) :
    start_time = time.time()
    l_sum = 0.0
    for x , y in data_iter:
      y = y.to(device); x = x.to(device)
      if check_flag:
          # print(f"y is {y}") # and y is {y}
          # logger.critical(f"y.shape is {y.shape} and x.shape is {x.shape} ") 
          # print(f"y.shape is {y.shape} and y.dtype is {y.dtype} ") # and y is {y}
          # print(f"x.shape is {x.shape} and x.dtype is {x.dtype} ")# and x is {x}
          # print(f"x.device is {x.device} and y.device is {y.device}")
          check_flag =False
          
      optimizer.zero_grad()  # ?? 自动帮我们清空计算图..
      y_ ,_= model(x,None)
      l = loss(y.float() ,y_).sum()

      # TODO 别人都是咋写的来着....
      # TODO 网络部分的可视化 专门开个函数来写吧
      if( False and epoch != 0):
        for tag,value in model.named_parameters():
            tag = tag.replace('.','/')
            writer.add_histogram(tag ,value.data ,epoch+1 )
            writer.add_histogram(tag+'/grad',value.grad.data,epoch+1 )

      nn.utils.clip_grad_norm_(model.parameters(), para("clip"))
      l.backward()
      l_sum += l.item()
      optimizer.step()                    # apply gradients
      # todo  attention here

    avg_loss =  l_sum / len(data_iter)
    logger.info("epoch %d, loss %.3f,elapsed time %.3f" \
      % ( epoch + 1, avg_loss ,time.time() - start_time ) )
    writer.add_scalar('loss',avg_loss ,epoch+1 )
    
    with open(f"{para.GetSaveDir()}/loss", 'a') as loss_file:
      loss_file.write(f"{epoch+1}\t{avg_loss}")

    torch.save(model.state_dict(), f"{para.GetSaveDir()}/latest_para.pt")
    if  avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), f"{para.GetSaveDir()}/min_loss_para.pt")

def test(model, dataset, para, device ):
  # 放在外面去
  loss = nn.MSELoss()
  check_flag = True
  model.eval() # TODO 实际上也没有正则化层啊...看看yuli代码...
  test_dataset = TrafficData( 
            dataset ,
            time_stamp = para("time_stamp"),
            test_index=(weekend,workday),
            train=False
            )
  
  # 一次把一整天取出来 如果不行就报错
  batch_size = 144//para("time_stamp")
  if batch_size != 144/para("time_stamp") :
    logger.error("test_dataset_iter error!!!")
    exit(-1)
  
  logger.info(f"batch_size(144//time_stamp) is { batch_size }")

  data_iter = Data.DataLoader(test_dataset, batch_size, shuffle=False,num_workers= para("num_workers") )
  model.load_state_dict(torch.load(f"{para.GetSaveDir()}/min_loss_para.pt"))
  
  data_iterator =iter(data_iter)
  # TODO 先取其中一天的出来看看   后面不仅要改成for循环  batch_size也要加大
  # for iter in tqdm( data_iter ):
  
  l_sum = 0.0
  for index,(x,y) in enumerate(data_iterator) : 
    # ( time_stamp , batch_size = 1 , 3 )   ( time_stamp , batch_size= 1 ,1 )
    pre_day ,true_day = [] , []
    y = y.to(device); x = x.to(device)
        # 这里还是可以再改进下的 比如说拿自己预测过了的放进去   或者teacher forcing
        # TODO 看看yul的teacher forcing如何实现
        # TODO ask老吴 是不是记录的时候也记录维度= =
    y_ , _ =  model( x ,None)

    # batch_size * time_stamp 
    pre_day = y_.detach().reshape(1,-1).squeeze() # .cpu().numpy()  # 
    true_day = y.detach().reshape(1,-1).squeeze() # .cpu().numpy()  #

    logger.info(f"true_day of {index//2} is {true_day.cpu().numpy()}")

    if check_flag:
      check_flag =False
      logger.critical(f"pre_day, true_day shape is {pre_day.shape} / {true_day.shape}" )

    l_sum += loss(pre_day,true_day)

    # save fig for comparation
    plt.plot(range(len(y)),pre_day.cpu().numpy())
    plt.plot(range(len(y_)),true_day.cpu().numpy())
    plt.legend( ["predict","ground_truth"] )
    plt.xlabel( "time step" )
    plt.ylabel( "traffic flow" )
    plt.title( f"predict {'weekend' if index%2==0 else 'weekday'} day of city {index//2}" )
    plt.savefig(f"test_{index//2}_{index%2}.svg")  # 0 for weekend and 1 for weekday

  logger.info(f"loss on test_set is {l_sum}")

# TODO pipeline在这里做就行
# test_dataset = TrafficData(test_data,time_stamp=time_stamp,device=device,test_index=(weekend,workday),train=False)


if __name__ == '__main__':
  rnn_regression = RegressionRnn(
                    input_dim =3, 
                    out_dim=1 ,
                    num_hidden = para("num_hidden"),
                    num_layer= para("num_layer"),
                    drop_rate = para("drop_rate") ).to(device)
  logger.debug(rnn_regression)

  # train(rnn_regression , train_data, para , writer , device )

  test( rnn_regression , test_data, para , device)

