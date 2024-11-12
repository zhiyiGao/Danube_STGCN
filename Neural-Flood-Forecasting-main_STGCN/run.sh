#!/bin/bash
#此参数用于指定运行作业的名称
#DSUB -n test

#DSUB -A root.dalhxwlyjsuo   
#默认参数，一般不需要修改  
#DSUB -l hws_gnode

#跨节点任务不同类型程序job_type会有差异，请参考下文对应跨节点任务模板编写
#DSUB --job_type cosched

#此参数用于指定资源。如申请 6核CPU，1卡GPU，48GB内存。
#DSUB -R 'cpu=12;gpu=2;mem=90000'

#此参数用于指定运行作业的机器数量。单节点作业则为 1 。
#DSUB -N 1

# 此参数用于指定日志的输出，%J表示JOB_ID。
#DSUB -e %J.out
#DSUB -o %J.out

#加载环境，此处加载anaconda环境以及通过anaconda创建的名为pytorch的环境(pytorch环境需要自己部署)
source  /home/dalhxwlyjsuo/criait_gaozy/anaconda3/etc/profile.d/conda.sh          
source activate GCN_Flood

#python 运行程序
python train_full.py
