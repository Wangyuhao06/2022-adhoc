import numpy as np

class Packet(object):
    def __init__(self,id_packet,packet_size,ori_node_id,dst_node_id,time_start_0):
        super(Packet, self).__init__()
        self.id=id_packet
        self.size=packet_size
        #节点属性
        self.ori_node_id=ori_node_id
        self.cur_node_id=ori_node_id
        self.dst_node_id=dst_node_id
        self.node_list=[ori_node_id]
        #T-T属性
        self.cur_trans_task_id=-100
        self.in_TR=0
        self.trans_task_IDlist=[]
        #路由属性
        self.time_start=time_start_0
        self.time_use=0
        self.arrive_flag=0
        
    def packet_trans_update(self,trans_task):
            if trans_task.trans_property[2]!=self.id:
                raise ValueError('trans_task not matched')
            self.cur_trans_task_id=trans_task.id
            
            