"""
    使用命令行 输入参数名称和对应值则会返回查找到的相应文件的文件名

    包含的参数与configs.config模块中的一致

    除此之外，还有hesy和dir的参数，前者用来测试，后者用于指定搜索目录，默认是当前路径
    
    详细可以输入-h参数进行查询
"""
import argparse
from configparser import ConfigParser

def find_file( file_dir ,argslist ):
    """ 查找包含输入列表的文件名 ，默认查找路径为当前目录
    Args
        argslist
            e.g. ["_time-stamp-12","_num_epochs-200" ] 
    Returns
        查找到的文件名,e.g. _time-stamp-12_num_epochs-200.10241201 , _time-stamp-12_lr-0.01_num_epochs-200.11301023
    """
    import os
    print("\t查询结果为：")
    for file in os.listdir(file_dir):
        include = True
        for n in argslist:
            if( n not in file ) :  include = False
            
        if include : print(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    cf = ConfigParser() 
    cf.read("config.ini") 
    for sec in cf.sections():
        for option in cf.options(sec):
            parser.add_argument( "--"+option)

    parser.add_argument( "-f" ,choices=[1,23], type= int ,help="这只是个choices选项的样例") # 如果不加 type = int 就会报错- - 因为解析出来的1是str类型...
    parser.add_argument( "-dir" ,default  ="./", help="默认查询当前目录" )
    parser.add_argument( "-hesy",help="这是个debug arg，用于测试"  )

    # 不传递参数 就是默认从命令行获取
    args = parser.parse_args()
    # 传递参数就会解析参数   当然 现在这个和上一个的对象是两个不一样的
    # args = parser.parse_args(['--foo', 'FOO'])

    # # 支持参数缩写 前缀匹配
    # args = parser.parse_args(["-he","1"])  但这里还没有成功....

    name  =[ f"_{key}-{value}" for (key,value) in vars(args).items() if value!= None and key!="dir" ]
    print(f"\t确认下输入的参数列表为：\n{name}")
    find_file(args.dir,name )