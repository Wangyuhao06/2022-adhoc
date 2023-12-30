import numpy as np
from src.parameter import *
class Trans_task(object):
    def __init__(self,trans_id,node_send,node_rec,packet):
        self.id=trans_id
        self.trans_property=(node_send.id,node_rec.id,packet.id)#基本属性
        self.packsize=packet.size
        ####frequency block info####
        self.FreqB_occup=node_send.current_freqB #占用频谱块id
        ####SINR and Capacity####
        self.SNR_C=([],1)#Y(SNR,Capacity)-----------------[X(timeslot1:SNR,Capacity),(timeslot2:SNR,Capacity),...]
        ####time of trans####
        self.time_use=1#int(self.packsize/self.SNR_C[1])+1
        self.time_cnt=0
        self.finish_flag=0
        ####energy setting####
        self.energy_property = (node_send.current_amp_send,RECAMP)
        self.energy_consume=(node_send.current_amp_send*packet.size*PACKENERGY,RECAMP*packet.size*PACKENERGY)
        self.power_consume=(round(node_send.current_amp_send*packet.size*PACKENERGY/self.time_use,6),round(RECAMP*packet.size*PACKENERGY/self.time_use,6))
    
    def show_info(self):
        return self.trans_property[0],self.trans_property[1],self.trans_property[2]
    
    def Trans_task_update(self):
        if self.finish_flag:
            return 1
        if self.time_cnt>=self.time_use:
            self.finish_flag=1
            return 1
        elif self.time_cnt<self.time_use:
            self.time_cnt+=1
            return 0
        
             
        #trans_task=tuple([],{},(node_send_id,node_send_amp,node_rec_id,node_rec_amp,packet_id),0)
        #tuple:([占用频谱块id]，{(timeslot1:SNR,Capacity),(timeslot2:SNR,Capacity),...},(基本属性:发送节点id,发送增益,接收节点id,接收增益,包id),完成标志位)