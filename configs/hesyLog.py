"""对项目的log进行基础的配置(格式) , 并创建hesy的logger 方便后续其他模块的logger进行继承"""
import logging
from colorlog import ColoredFormatter
import sys
import os

logger = None
 
LOGFORMAT = " %(levelname)-8s  - %(filename)s - %(funcName)s - %(log_color)s%(asctime)s| %(message)s%(reset)s"
formatter = ColoredFormatter(LOGFORMAT)

logging.basicConfig(datefmt='%m/%d-%H:%M',level=logging.DEBUG,
    format = "{%(levelname)-4s}-[%(filename)s]-[%(funcName)s]-[%(asctime)s]| %(message)s")
# lazy load
def LogTo( logDir ):
    """绑定一个stream handler和一个file handler到hesy logger上
        
        info等级的信息输出到console, debug等级的信息输出到log文件中

        args
            logDir 
                filehander记录的信息存储的目录
                存储的文件是 logDir/Filehandler.log
    """
    global logger
    logger = logging.getLogger("hesy")
    # TODO record  https://zhuanlan.zhihu.com/p/56095714
    logger.propagate = False
    # TODO 在这里设置datefmt
    LOGFORMAT = "%(log_color)s{%(levelname)-4s}-[%(filename)s]-[%(funcName)s]-[%(asctime)s]| %(message)s%(reset)s"
    formatter = ColoredFormatter(LOGFORMAT)
    '''一般就是debug信息输入到文件里面  然后info信息提示进行到哪里了 ''' 
    # TODO 添加 rotation size 以及最大的文件数量...
    file = logDir + "/Filehandler.log"

    if not os.path.exists( logDir):
        os.makedirs( logDir ) 

    with open( file ,mode = 'a'):
        pass
 
    fh = logging.FileHandler( filename = file ) # 默认就是mode ="a"
    sh = logging.StreamHandler(sys.stdout)
    # sh = logging.StreamHandler()
    sh.setLevel( logging.INFO )
    fh.setLevel( logging.DEBUG )
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    # return logger
    logger.info(f"Save log file into {file}")  

# TODO
"""
整理下filter的用法  https://segmentfault.com/a/1190000003008066
.gitignore也放到脚手架里面
yaml配置logging
traditional logging https://docs.python.org/zh-cn/3/howto/logging-cookbook.html
requirements.txt ( pip frozen ? )
可以抄下别人的jupyer记录的latex语法 然后放到zepto上
ask老吴同步参数（ 其实不用ask他 自己找自己整理下就知道了
去GitHub上看看有咩有wrapper python log

只有在同一个包内才能用相对路径导包
http://www.pythondoc.com/pythontutorial3/modules.html 这个是中文文档 挺好的  马
    * 模块也可以包含可执行语句。这些语句一般用来初始化模块。
        他们仅在 第一次 被导入的地方执行一次
        修改了代码以后需要reload
    * from xx import *  可以导入所有除了以下划线( _ )开头的命名 --》 看来如果不想被导入 就写成 _ 开头的就好  或者使用__all__ 变量控制export的变量
        但实际上不是一很鼓励--》 代码比较难以阅读

"""