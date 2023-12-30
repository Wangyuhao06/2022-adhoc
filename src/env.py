import random
from math import log2, log10
from queue import Queue

import numpy as np

from pymobility.models.mobility import random_waypoint
from src.node import Node
from src.packet import Packet
from src.parameter import *
from src.transtask import Trans_task


class Environment():
     #初始化环境
    def __init__(self):
        #初始数据-最大节点数
        self.node_max=NODE_MAX
        self.node_space_size=NODE_MAX
        self.node_moving_area=MOV_AREA
        #初始化二维平面
        self.geo_area = random_waypoint(self.node_max, dimensions=(MOV_AREA, MOV_AREA), velocity=(10, 15), wt_max=1.0)
        self.position=0
        #初始化随机相邻矩阵
        self.topology = np.zeros((self.node_space_size,self.node_space_size))
        self.topology[0:self.node_max,0:self.node_max] = np.random.randint(0,2,(self.node_max,self.node_max))
        for i in range(self.node_max):
            self.topology[i,i] = 1
            for j in range(self.node_max):
                #构建双向图
                if self.topology[i,j] == 1:
                    self.topology[j,i] = 1
        #初始化节点动作空间
        self.topology_actSpace=[]
        #初始化频谱块元组-----(0,[])表示(占用与否,[占用transtaskID列表]) 
        self.freqB_list=([],[],[],[],[],[],[],[],[],[]) #((0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]))
        self.freqB_use_history=([],[],[],[],[],[],[],[],[],[])
        #初始化传输事件列表
        self.trans_task_ID_inTR=[]
        self.trans_task_list=[]
        self.trans_task_cnt=0 # id计数器
        #初始化包列表
        self.amount_poisson_list = np.random.poisson(lam=LAMDA,size=MAX_TIME)#包数量初始化
        self.size_normal_list = ((np.random.normal(0,1,MAX_TIME*2)*16+16)//8)*8#包大小初始化
        self.pack_use_cnt=0#包序号计数器
        self.packets_list=[]#包列表
        self.packets_live_id=[]
        #初始化节点列表
        self.node_list=[]
        self.live_node_ID_list=[]
        for i in range(self.node_max):
            locals()['node_'+str(i)] = Node(i)
            self.node_list.append(locals()['node_'+str(i)])
            self.live_node_ID_list.append(i)
        #噪声系数
        self.noise_list = np.random.rayleigh(1,MAX_TIME*2)#*NOISE_CONST/2
        #统计参数
        self.envTr_time=0
        self.allNode_pw=0
        self.allNode_delay=0
        self.time_avg=0
        self.arrive_time=1
        self.end=0
        self.terminate=0
        
        self.packet_arrive_success=[]
        self.agent_arrive=[]
        for i in range(NODE_MAX):
            self.packet_arrive_success.append(0)#节点作为 |源节点| 发的包成功到达数
            self.agent_arrive.append(0)#节点作为 |最后一个中间节点| 发的包成功到达数
        # self.sum_packet_done_rate=0
        #四元组
        self.all_ob=np.array([[0]*OBS_LEN]*NODE_MAX)
        self.reward=np.array([1]*self.node_max)
        self.para_reward=np.array([1]*self.node_max) 
    
    def generate_packet(self,cur_time):
        packetsList_temp=[]
        packets_cnt=self.amount_poisson_list[cur_time]
        for i in range(packets_cnt):
            nodes_temp = random.sample(self.live_node_ID_list,2)
            locals()['packet_'+str(self.pack_use_cnt)]=Packet(self.pack_use_cnt,abs(self.size_normal_list[self.pack_use_cnt])+8,nodes_temp[0],nodes_temp[1],cur_time)
            self.packets_list.append(locals()['packet_'+str(self.pack_use_cnt)])
            self.packets_live_id.append(self.pack_use_cnt)
            packetsList_temp.append(locals()['packet_'+str(self.pack_use_cnt)])
            self.node_list[nodes_temp[0]].packets_ToSend_id.append(self.pack_use_cnt)
            self.node_list[nodes_temp[0]].packets_id_list.append(self.pack_use_cnt)
            self.pack_use_cnt+=1
        return packetsList_temp
    
    #传输任务更新
    def trans_task_update(self,cur_time):
        
        if len(self.trans_task_ID_inTR)>0 and len(self.trans_task_list)>0:
            #所有在传传输任务
            for trans_temp_id in self.trans_task_ID_inTR:
                task_finish=self.trans_task_list[trans_temp_id].Trans_task_update()
                node_send_id,node_rec_id,packet_id=self.trans_task_list[trans_temp_id].show_info()
                #包传输更新
                self.packets_list[packet_id].time_use+=1
                #节点更新
                # self.node_list[node_send_id].next_hop_id=node_rec_id
                if node_send_id!=node_rec_id:   
                    self.node_list[node_send_id].power_list.append(self.trans_task_list[trans_temp_id].power_consume[0])
                    self.node_list[node_send_id].current_power_send=self.trans_task_list[trans_temp_id].power_consume[0]
                    self.node_list[node_send_id].energy_consumption+=self.trans_task_list[trans_temp_id].power_consume[0]

                    self.node_list[node_rec_id].power_list.append(self.trans_task_list[trans_temp_id].power_consume[1])
                    self.node_list[node_rec_id].current_power_receive=self.trans_task_list[trans_temp_id].power_consume[1]
                    self.node_list[node_rec_id].energy_consumption+=self.trans_task_list[trans_temp_id].power_consume[1]
                #统计参数更新
                self.envTr_time+=1
                
                #trans任务完成更新
                if task_finish and self.topology[node_send_id,node_rec_id]==1 :
                    #更新包与节点
                        # T-T清除
                    self.trans_task_ID_inTR.remove(trans_temp_id)
                        # 包属性清除
                    self.packets_list[packet_id].in_TR=0
                    self.packets_list[packet_id].cur_trans_task_id=0
                    self.packets_list[packet_id].cur_node_id=node_rec_id
                        # 发送节点属性清除
                    self.node_list[node_send_id].packets_ToSend_id.remove(packet_id)
                    self.node_list[node_send_id].trans_task_send.get()
                    self.node_list[node_send_id].sending_flag=0
                    self.node_list[node_send_id].current_amp_send=0
                    self.node_list[node_send_id].current_power_send=0
                        # 接收节点属性清除
                    self.node_list[node_rec_id].trans_taskID_rec.remove(trans_temp_id)
                    if len(self.node_list[node_rec_id].trans_taskID_rec)==0:
                        self.node_list[node_rec_id].rec_flag=0
                    # self.node_list[node_rec_id].current_amp_receive=0
                    self.node_list[node_rec_id].current_power_receive=0
                        # 频谱环境更新(频谱块release)
                    freqB_ID_now=0
                    for freqB_ocp_now in self.trans_task_list[trans_temp_id].FreqB_occup:
                        if freqB_ocp_now and node_send_id!=node_rec_id:
                            self.freqB_list[freqB_ID_now].remove(node_send_id)
                        freqB_ID_now+=1

                    #判断是否到达目的地   
                    if self.packets_list[packet_id].cur_node_id==self.packets_list[packet_id].dst_node_id and self.topology[node_send_id,node_rec_id]==1:
                        # 可通信到达
                        self.packets_list[packet_id].arrive_flag=1
                        self.packets_live_id.remove(packet_id)
                        ### 记录接受节点和发出节点的奖励 ###
                        self.packet_arrive_success[self.packets_list[packet_id].ori_node_id]+=1
                        self.agent_arrive[node_send_id]+=1 
                        # self.arrive_time += self.trans_task_list[trans_temp_id].time_use # datacheck3
                        self.arrive_success += 1
                    elif self.topology[node_send_id,node_rec_id]==1 :
                        #可通信没到达
                        self.node_list[node_rec_id].packets_ToSend_id.append(packet_id)
                        # self.arrive_time += (cur_time - self.packets_list[packet_id].time_start) # datacheck3
                    else:
                        #不可通信
                        self.trans_task_list[trans_temp_id].time_cnt=0
                        self.trans_task_list[trans_temp_id].finish_flag=0
        # for packet_id in self.packets_live_id:
        #     #判断是否到达目的地   
        #     if self.packets_list[packet_id].cur_node_id==self.packets_list[packet_id].dst_node_id or self.packets_list[packet_id].arrive_flag==1:
        #         #到达
        #         continue
        #         # self.arrive_time += self.trans_task_list[trans_temp_id].time_use
        #     else:#没到达
        #         self.arrive_time += 1
        self.arrive_time += len(self.packets_live_id)
                
                     
                
    def all_agent_observe(self):    
        all_ob=[]
        # fBlst=[0,0,0,0,0,0,0,0,0,0]
        degree=0
        pack_storage=0
        pw_avg_all=0
        dst_node=-1

        # for node_id in range(self.node_max):
        #     if len (self.node_list[node_id].packets_ToSend_id):
        #         packet_toSend_id=self.node_list[node_id].packets_ToSend_id[0]
        #         dst_node=self.packets_list[packet_toSend_id].dst_node_id
                
        #     else:
        #         dst_node=-1
            
        # for node_id in self.live_node_ID_list:
        # for node_id in range(self.node_max):
        #     fb_tp_id=0
        #     for fb_tp in self.node_list[node_id].current_freqB:
        #         fBlst[fb_tp_id]=fb_tp
        #         fb_tp_id+=1
        
        # for node_id in self.live_node_ID_list:
            #neibor_idlist=self.node_list[node_id].neibor_idlist[:]#深复制
            #receive ob?
            #neibor_idlist.append(node_id)
            #neibor_vector=[]
            #for i in neibor_idlist:
        
        # for node_id in range(self.node_max):
        #     pwl=self.node_list[node_id].power_list
        #     if len(pwl)>=BACKTIME:
        #         pwlst=pwl[len(pwl)-BACKTIME:len(pwl)]
        #     else:
        #         pwlst=pwl    
        #     if len(pwlst)>0:
        #         pw_avg=sum(pwlst)/len(pwlst)
        #     else:
        #         pw_avg=0
        #     pw_avg_all+=pw_avg
        
        for node_id in range(self.node_max):
            pwl=self.node_list[node_id].power_list
            if len(pwl)>=BACKTIME:
                pwlst=pwl[len(pwl)-BACKTIME:len(pwl)]
            else:
                pwlst=pwl    
            if len(pwlst)>0:
                pw_avg=sum(pwlst)/len(pwlst)
            else:
                pw_avg=0
            
            if len (self.node_list[node_id].packets_ToSend_id)>0:
                packet_toSend_id=self.node_list[node_id].packets_ToSend_id[0]
                dst_node=self.packets_list[packet_toSend_id].dst_node_id  
            else:
                dst_node=-1
            
            pw=[]
            pw.append(pw_avg)
            
            dgr=[]
            degree=len(self.topology_actSpace[node_id][0])-1
            dgr.append(degree)
            
            pcs=[]
            pack_storage=len(self.node_list[node_id].packets_ToSend_id)
            pcs.append(pack_storage)
            
            dn=[]
            dn.append(dst_node)
            
            all_ob.append(pw+dgr+pcs+dn)
            #self.node_list[node_id].ob_send=neibor_vector
       
        return np.array(all_ob)
        
    
    # def generate_trans_task(self,trans_id,send_node,rec_node,packet):
    #     trans_task_temp=Trans_task(trans_id,send_node,rec_node,packet)
    #     return trans_task_temp
        
    def env_check_right(self):
        for node_id in self.live_node_ID_list:
            if self.node_list[node_id].trans_task_send.empty():
                assert self.node_list[node_id].sending_flag == 0
            elif not self.node_list[node_id].trans_task_send.empty():
                assert self.node_list[node_id].sending_flag == 1
                st_temp=self.node_list[node_id].trans_task_send.get()
                self.node_list[node_id].trans_task_send.put(st_temp)#无损使用队列内容
                s_node_send_id,s_node_rec_id,s_packet_id=st_temp.show_info()
                assert node_id==s_node_send_id
                # assert self.node_list[node_id].next_hop_id==s_node_rec_id
                assert self.node_list[node_id].packets_ToSend_id[0]==s_packet_id
                
            elif self.node_list[node_id].trans_task_rec.empty():
                assert self.rec_flag == 0
            elif not self.node_list[node_id].trans_task_rec.empty():
                assert self.node_list[node_id].rec_flag == 1
                rt_temp=self.node_list[node_id].trans_task_rec.get()
                self.node_list[node_id].trans_task_rec.put(rt_temp)#无损使用队列内容
                r_node_send_id,r_node_rec_id,r_packet_id=rt_temp.show_info()
                assert node_id==r_node_rec_id
                # assert self.node_list[node_id].next_hop_id==s_node_rec_id
                assert self.node_list[node_id].packets_ToSend_id[0] != r_packet_id
 
        return 0    
   

    def topology_update(self,cur_time,rand_change):
        self.topology = np.zeros((NODE_MAX,NODE_MAX))
    ################--------随机更改拓扑结构--------################
        if rand_change:
            positions=next(self.geo_area)
            self.position = positions
            for a in range(NODE_MAX):
                for b in range(NODE_MAX):
                    if np.linalg.norm(positions[a]-positions[b]) <= COM_RANGE:
                        self.topology[a,b]=1
                        self.topology[b,a]=1
                    else:
                        self.topology[a,b]=0
                        self.topology[b,a]=0
            # if np.random.rand()<DELTA and cur_time%30==0:
            #     for i in np.random.randint(0,self.node_max,np.random.randint(3)+1):
            #         self.topology[i,:]=np.random.randint(0,2,self.node_max)
            #         self.topology[i,i] = 1
            #         for j in range(self.node_max):
            #             #构建双向图
            #             if self.topology[i,j] == 1:
            #                 self.topology[j,i] = 1
        # print(positions)
        # print("****************")
        # print(self.topology)
        # print("------------------------------------")
      ################--------更新邻域--------################
        self.live_node_ID_list=[]
        self.topology_actSpace=[]
        for i in range(self.topology.shape[0]):
            if any(self.topology[i,:]):
                TPtemp = np.nonzero(self.topology[i,:])
                # self.node_list[i].neibor_idlist=TPtemp
                self.topology_actSpace.append(TPtemp)
                self.live_node_ID_list.append(i)
            else:
                TPtemp = -1
                self.topology_actSpace.append(TPtemp)
        return self.topology
   
      
    def get_state_reward(self):
        
        return self.topology,self.all_ob,self.reward     
    
    def time_step(self,cur_time,action):
        
        self.packet_arrive_success=[]
        self.agent_arrive=[]
        for i in range(NODE_MAX):
            self.packet_arrive_success.append(0)
            self.agent_arrive.append(0)
        self.arrive_success=0
        
        # self.env_check_right()
        topology_now=self.topology_update(cur_time,1)
        self.generate_packet(cur_time)
        self.all_ob=self.all_agent_observe()
        self.trans_task_update(cur_time)
        for node_index in self.live_node_ID_list :
            if  len(self.node_list[node_index].packets_ToSend_id)>0 and self.node_list[node_index].sending_flag!=1:
                packet_toSend_id=self.node_list[node_index].packets_ToSend_id[0]
                #包未到达且非在传----->生成trans_task
                if self.packets_list[packet_toSend_id].arrive_flag==0 and self.packets_list[packet_toSend_id].in_TR==0:
                    #传输和接收节点决策
                    send_node=self.node_list[node_index]
                    Action=action[node_index]#######################################################
                    next_hop_id,current_freqB,current_amp_send=Action[0],Action[1:N_ACTION_C],Action[N_ACTION_C]
                    send_node.next_hop_id=next_hop_id       
                    rec_node=self.node_list[next_hop_id]
                    current_amp_rec=RECAMP
                    
                    self.node_list[node_index].current_freqB=current_freqB
                    self.node_list[node_index].next_hop_id=next_hop_id
                    self.node_list[node_index].current_amp_send=current_amp_send
                    #频谱环境更新
                    freqB_ID_now=0
                    for fB_ocp in current_freqB:
                        if node_index!=next_hop_id and fB_ocp:
                            self.freqB_list[freqB_ID_now].append(node_index)
                            self.freqB_use_history[freqB_ID_now].append(node_index)
                        freqB_ID_now+=1
                    #T-T生成与T-T环境更新
                    trans_task_now=Trans_task(self.trans_task_cnt,send_node,rec_node,self.packets_list[packet_toSend_id])
                    trans_task_now.SNR_C=self.SNR_cac_update(cur_time,trans_task_now,current_amp_send,current_freqB,current_amp_rec)
                    trans_task_now.time_use=int(trans_task_now.packsize/(trans_task_now.SNR_C[1]))+1
                    
                    if node_index==next_hop_id:
                        trans_task_now.time_use=1#节点内部等待
                        
                        #节点与包写入
                            #发送节点任务、标志更新
                    self.node_list[node_index].trans_task_send.put_nowait(trans_task_now)
                    self.node_list[node_index].sending_flag=1
                            #接收节点任务、标志更新
                    self.node_list[next_hop_id].trans_taskID_rec.append(trans_task_now.id)
                    self.node_list[next_hop_id].rec_flag=1
                            #包任务、标志更新
                    self.packets_list[packet_toSend_id].cur_trans_task_id=self.trans_task_cnt
                    self.packets_list[packet_toSend_id].in_TR=1
                        #T-T环境写入
                    self.trans_task_ID_inTR.append(trans_task_now.id)
                    self.trans_task_list.append(trans_task_now)
                    self.trans_task_cnt+=1
        #reward清算
            #总传输时间为self.envTr_time，总时间为cur_time
        packet_done_rate=1-round((len(self.packets_live_id)+0.1)/(len(self.packets_list)+0.1),4)#包传输完成率为packet_done_rate
        # self.avg_packet_done_rate += packet_done_rate
        # self.time_avg+=self.envTr_time/(1+len(self.packets_list)-len(self.packets_live_id))
        # self.time_avg+=packet_done_rate
        # self.time_avg+=self.arrive_time/((1+len(self.packets_list)-len(self.packets_live_id))*(packet_done_rate))
        
        # print("pdr: "+str(packet_done_rate))
        if packet_done_rate<=0.03:
            packet_done_rate=0.03
            
        self.time_avg+=self.arrive_time/(1+len(self.packets_list)-len(self.packets_live_id))#*(packet_done_rate)
        
        # if len(self.packets_live_id) == 0:
        #     self.terminate=1
        if len(self.packets_live_id) == 0:
            self.end=1
            
        for i in range(self.node_max):
            # pw_sum=sum(self.node_list[i].power_list)
            if len(self.node_list[i].power_list)>0:
                pw_now=self.node_list[i].power_list[-1]+1
            else:
                pw_now=1
            if not self.node_list[i].trans_task_send.empty():
                st_temp=self.node_list[i].trans_task_send.get()
                self.node_list[i].trans_task_send.put_nowait(st_temp)#无损使用队列内容
                trans_delay=st_temp.time_use+1
            else:
                trans_delay=1
                
            self.para_reward[i]= -trans_delay*pw_now
            # if packet_done_rate<=0.05:
            #     packet_done_rate=0.05
            # self.reward[i]=round((packet_done_rate*cur_time*cur_time*self.node_max*DEVICE_ENERGY+0.0001)/(pw_sum*self.envTr_time+0.0001),6)
            # self.reward[i]=round((packet_done_rate*cur_time*self.node_max*DEVICE_ENERGY+1000)/(pw_sum+1000),6)
            # self.reward[i]=round(-(pw_sum*self.envTr_time)/((packet_done_rate+0.1)*(cur_time+1)*(cur_time+1)*self.node_max),6)#RWD2
            # self.reward[i]=round(-(10*pw_sum*self.envTr_time)/((packet_done_rate)*(cur_time+1)*(cur_time+1)*self.node_max),6)#RWD3
            # self.reward[i]=round(-(pw_sum*self.envTr_time)/((packet_done_rate+0.001)*(cur_time+1)*(cur_time+1)*self.node_max),6)
            # self.reward[i]=round(-(10*pw_sum*self.envTr_time)/((packet_done_rate)*(cur_time+1)*(cur_time+1)*self.node_max),6)#RWD3
            #self.reward[i]=round(-(10*log10(10+pw_sum)*self.arrive_time)/((1+len(self.packets_list)-len(self.packets_live_id))*(packet_done_rate)*(cur_time+1)*self.node_max),6)#RWD4
            # self.reward[i]=round(-(10*pw_sum*self.arrive_time)/((1+len(self.packets_list)-len(self.packets_live_id))*(packet_done_rate)*(cur_time+1)*self.node_max),6)#RWD5
            # self.reward[i]=-(10*log10(10+pw_sum)*(self.arrive_time+1))/((1+len(self.packets_list))*(packet_done_rate)*(cur_time+1)*self.node_max)#RWD6
            # self.reward[i]=self.arrive_success*1000-(self.arrive_time/(1+len(self.packets_list))*log10(10+pw_sum/(cur_time+1)))#RWD7
            # self.reward[i]=self.terminate*10000+self.arrive_success*300-100*self.arrive_time/(1+len(self.packets_live_id))/len(self.packets_list)#RWD8
            # self.reward[i]=(self.terminate*10000+self.arrive_success*1000-len(self.packets_live_id))/self.node_max#RWD9
            
            # ###RWD10###
            # if self.agent_arrive[i]==0 and self.packet_arrive_success[i]==0 and self.terminate!=1 :
            #     self.reward[i] = - len(self.node_list[i].packets_ToSend_id) + 10000*self.end#*(1-self.terminate)  #等待时延
            # elif self.agent_arrive[i]>0 or self.packet_arrive_success[i]>0 and self.terminate!=1:
            #     # self.reward[i]=1000*self.agent_arrive[i]+1000*self.packet_arrive_success[i]+10000*self.terminate
            #     self.reward[i] = 1000*self.agent_arrive[i] + 10000*self.end#*(1-self.terminate) 
            # elif self.terminate:
            #     self.reward[i] = 2000
            # # self.reward[i] = 1000*self.agent_arrive[i] + 1000*self.packet_arrive_success[i] -len(self.node_list[i].packets_ToSend_id)
            # ###########
            
            ###RWD11###
            if self.agent_arrive[i]==0 and self.packet_arrive_success[i]==0 :
                self.reward[i] = (- len(self.node_list[i].packets_ToSend_id) - 100*trans_delay) #等待时延 +  传输时延
            elif self.agent_arrive[i]>0 or self.packet_arrive_success[i]>0 :
                self.reward[i] = 1000*self.agent_arrive[i] #+ self.para_reward[i] #本节点作为 |最后一个中间节点| 发的包成功到达数 
            ###########
            
            self.allNode_delay+=trans_delay
            self.allNode_pw+=round(pw_now,6)
            
        if len(self.packets_live_id) == 0:
            self.terminate=1
        
        
        # self.time_avg+=self.envTr_time/(1+len(self.packets_list)-len(self.packets_live_id))*(cur_time+1)
        print("pdr: "+str(packet_done_rate)+"   rwd: "+str(sum(self.reward)))
        return topology_now,self.all_ob,self.reward,self.para_reward,self.terminate

        
        
    def SNR_cac_update(self,cur_time,trans_task,current_amp_send,current_freqB,current_amp_rec):
        trans_task_temp=trans_task
        node_send_id,node_rec_id,packet_id=trans_task_temp.show_info()
        trans_energy=round(current_amp_send*current_amp_rec*self.packets_list[packet_id].size*PACKENERGY,6)
        noise=NOISE_CONST
        SINR_fB=[]
        Capacity=1
        node_range = np.linalg.norm(self.position[node_send_id]-self.position[node_rec_id])
        for fB_id in range(len(current_freqB)):
            inference_temp=noise
            if current_freqB[fB_id]:
                node_list_temp=self.freqB_list[fB_id]
                for i in node_list_temp:
                    if i==node_send_id:
                        continue
                    elif i in self.topology_actSpace[node_rec_id][0]:
                        if self.node_list[i].sending_flag==1:
                            ts_ttemp=self.node_list[i].trans_task_send.get_nowait()
                            oth_node_send_id,oth_node_rec_id,oth_packet_id=ts_ttemp.show_info()
                            inference_temp+=round(self.node_list[oth_node_send_id].current_amp_send*RECAMP*self.packets_list[oth_packet_id].size*PACKENERGY,6)
                            self.node_list[i].trans_task_send.put_nowait(ts_ttemp)#无损使用队列内容
                Sinr=round(trans_energy*(10**(-abs(self.noise_list[cur_time])))*10**(-node_range/100/COM_RANGE)/inference_temp,6)
                Capacity+=round(8*4*log2(1+Sinr),6)
                SINR_fB.append(Sinr)
        return (SINR_fB,Capacity)
          
                          
    def reset(self):
        #初始化二维平面
        self.geo_area = random_waypoint(self.node_max, dimensions=(MOV_AREA, MOV_AREA), velocity=(10, 15), wt_max=1.0)
        #初始化随机相邻矩阵
        self.topology = np.zeros((self.node_space_size,self.node_space_size))
        self.topology[0:self.node_max,0:self.node_max] = np.random.randint(0,2,(self.node_max,self.node_max))
        for i in range(self.node_max):
            self.topology[i,i] = 1
            for j in range(self.node_max):
                #构建双向图
                if self.topology[i,j] == 1:
                    self.topology[j,i] = 1
        #初始化节点动作空间
        self.topology_actSpace=[]
        #初始化频谱块元组-----(0,[])表示(占用与否,[占用transtaskID列表]) 
        self.freqB_list=([],[],[],[],[],[],[],[],[],[]) #((0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]),(0,[]))
        self.freqB_use_history=([],[],[],[],[],[],[],[],[],[])
        #初始化传输事件列表
        self.trans_task_ID_inTR=[]
        self.trans_task_list=[]
        self.trans_task_cnt=0 # id计数器
        #初始化包列表
        self.amount_poisson_list = np.random.poisson(lam=LAMDA,size=MAX_TIME)#包数量初始化
        self.size_normal_list = ((np.random.normal(0,1,MAX_TIME*2)*16+16)//8)*8#包大小初始化
        self.pack_use_cnt=0#包序号计数器
        self.packets_list=[]#包列表
        self.packets_live_id=[]
        #初始化节点列表
        self.node_list=[]
        self.live_node_ID_list=[]
        for i in range(self.node_max):
            locals()['node_'+str(i)] = Node(i)
            self.node_list.append(locals()['node_'+str(i)])
            self.live_node_ID_list.append(i)
        #统计参数
        self.envTr_time=0
        self.allNode_pw=0
        self.allNode_delay=0
        self.time_avg=0
        self.arrive_time=1
        # self.arrive_success=0
        self.terminate=0
        self.end=0
        self.packet_arrive_success=[]
        self.agent_arrive=[]
        for i in range(NODE_MAX):
            self.packet_arrive_success.append(0)
            self.agent_arrive.append(0)
        #四元组
        self.all_ob=np.array([[0]*OBS_LEN]*NODE_MAX)
        self.reward=np.array([1]*self.node_max)        
                          
    