import numpy as np
from queue import Queue

class Node(object):
    def __init__(self,id_node):
        super(Node, self).__init__()
        #multi-agent sys setting
        self.node_max=36
        self.act_range=self.node_max-1 #最大邻居范围
        # current agent-property setting
        self.id=id_node#该节点id
        # 1 - packets
        self.packets_ToSend_id=[]#该节点当前待传的包
        self.packets_id_list=[]#该节点至今为止保存过的包id
        
        self.sending_flag=0
        self.rec_flag=0
        
        self.trans_task_send=Queue(maxsize=1)#该节点当前传输的任务
        self.trans_taskID_rec=[]#该节点当前接收的任务
        # 2 - energy
        self.current_amp_send=0#节点当前发送增益--------动作
        #self.current_amp_receive=0#节点当前接收增益--------动作
        
        self.current_power_send=0#节点当前发送功率
        self.current_power_receive=0#节点当前接收功率
        self.power_list=[]#节点使用能量记录
        
        self.energy_consumption=0#截至现在能量消耗
        # 3 - freq
        self.current_freqB=[1]#当前选用频谱块--------动作
        self.freqB_list=[1]#频谱块历史
        # 4 - topology
        self.neibor_idlist=[]
        self.next_hop_id=-1#下一条节点id--------动作
        # 5 - observation
        #self.ob_send=[]
    
    # def observation_rec(self,send_node):
    #     if len(self.ob_send)==0 or len(send_node.ob_send)==0 :
    #         raise ValueError("send observation unfinished")
    #     self.ob_rec.append(self.ob_send[-1])
    #     self.ob_rec.append(send_node.ob_send[-1])
    #     return self.ob_rec
        
        
    def get_send_action(self,ob,action_space):
        
        ###缺省决策###
        
        #改变属性
        return self.current_amp_send,self.current_freqB,self.next_hop_id
            
    def get_rec_action(self,ob):
        
        ###缺省决策###
        
        #改变属性
        return self.current_amp_receive   