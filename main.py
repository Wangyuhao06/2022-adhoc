from src.env import Environment
from src.node import Node
from src.packet import Packet
from src.transtask import Trans_task
from src.DGN import DGN,DPG
from src.parameter import *
from src.buffereplay import ReplayBuffer
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import numpy as np
from queue import Queue
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

env=Environment()
buff=ReplayBuffer(BUFFERSIZE)

model = DGN(NODE_MAX,OBS_LEN,HIDDEN_DIM,N_ACTION_D)
model_tar = DGN(NODE_MAX,OBS_LEN,HIDDEN_DIM,N_ACTION_D)
# para_model=DPG(NODE_MAX,OBS_LEN,HIDDEN_DIM,N_ACTION_C)
# para_model_tar=DPG(NODE_MAX,OBS_LEN,HIDDEN_DIM,N_ACTION_C)
model = model.cuda()
model_tar = model_tar.cuda()
# para_model=para_model.cuda()
# para_model_tar=para_model_tar.cuda()
# model = model.cuda()
# model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# para_optim = optim.Adam(para_model.parameters(), lr = 0.1)

O = np.ones((BATCH_SIZE,NODE_MAX,OBS_LEN))
Next_O = np.ones((BATCH_SIZE,NODE_MAX,OBS_LEN))
A_d = np.ones((BATCH_SIZE,NODE_MAX,12))
Matrix = np.ones((BATCH_SIZE,NODE_MAX,NODE_MAX))
Next_Matrix = np.ones((BATCH_SIZE,NODE_MAX,NODE_MAX))
onehot = np.eye(NODE_MAX)


list_mul = lambda x,y:x*y

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

score = 0
f = open('rwd.txt','w+')
f2 = open('pw.txt','w+')
f3 = open('time.txt','w+')
f4 = open('loss.txt','w+')
f5 = open('para_loss.txt','w+')

i_episode=0

while i_episode<20000:#N_EPISODE:

	if i_episode > 10:
		epsilon -= 0.04
		if epsilon < 0.1:
			epsilon = 0.1
	i_episode+=1
	TimeSteps = 0
	# paraLoss_tar = 0
 
	all_time_AVGpw=0
	all_time_AVGtrtime=0
	env.reset()
    
	# for i in range(500):
	# 	env.generate_packet(i)
     
	while TimeSteps < MAX_STEP:
		TimeSteps += 1 
		action=[]
		topology,all_ob,reward = env.get_state_reward()
		# para=para_model( torch.Tensor(np.array([all_ob])).cuda() )
		# q = model(torch.Tensor(np.concatenate((np.array([all_ob]),para.detach().numpy()), axis=np.array([all_ob]).ndim-1)), torch.Tensor(topology))[0]
		with torch.no_grad():
			q0 = model(torch.Tensor(np.array([all_ob])).cuda(), torch.Tensor(topology).cuda())[0]
			# q=q.detach().numpy()
			q=q0.detach().cpu().numpy()
		i=0
		# para_use=para[0]
		for i in range(NODE_MAX):
			# #continuous action setting
			a_c=[]
			a_c.append(0)
			a_c=a_c*N_ACTION_C
			for j in range(N_ACTION_C-1):
				a_c[j] = 1
			a_c[N_ACTION_C-1] = 10**(100/1000)
			
			# a_c=[]
			# a_c.append(0)
			# a_c=a_c*N_ACTION_C
			# if np.random.rand() < epsilon:
			# 	for j in range(N_ACTION_C-1):
			# 		a_c[j] = np.random.randint(2)
			# 	a_c[N_ACTION_C-1]=pow(10,np.random.randint(3))
				
			# else:
			# 	para_flag=0
			# 	for j in range(N_ACTION_C-1):
			# 		if para_use[i][j]>0:
			# 			a_c[j] = 1
			# 			para_flag=1
			# 	if not para_flag:
			# 		a_c[np.random.randint(10)]=1
				
			# 	if para_use[i][N_ACTION_C-1]<=1 and para_use[i][N_ACTION_C-1]>0.333:
			# 		a_c[N_ACTION_C-1] = 100
			# 	elif para_use[i][N_ACTION_C-1]>=-0.333 and para_use[i][N_ACTION_C-1]<=0.333:
			# 		a_c[N_ACTION_C-1] = 10
			# 	elif para_use[i][N_ACTION_C-1]>=-1 and para_use[i][N_ACTION_C-1]<-0.333:
			# 		a_c[N_ACTION_C-1] = 1

			#discrete action setting
			a_d=[]
			a_d.append(0)
			if np.random.rand() < epsilon:
				a_d[0] = np.random.randint(N_ACTION_D)
			else:
				q[i]=list(map(list_mul,q[i],topology[i,:]))
				q[i,:]=q[i,:]-(1-topology[i,:])*100000.0
				q[i][i]=-90000.0
				a_d[0] = q[i].argmax().item()
				if not (topology[i,:]-onehot[i,:]).any():
					a_d[0]=i
        
			action.append(a_d+a_c)
		topology_next, all_ob_next,reward_next,para_reward,terminate = env.time_step(TimeSteps,action)
		buff.add(np.array(all_ob),action,reward,np.array(all_ob_next),topology,topology_next,para_reward,terminate)
		all_ob = all_ob_next
		topology = topology_next
		score += sum(reward/100)
		# paraLoss_tar += sum(reward/36000)
  
  ##############################
		# if TimeSteps <= BATCH_SIZE or TimeSteps%50 != 0:
		#  	continue
		# if i_episode < 10:
		# 	continue

		# for e in range(N_EPOCH):
			
		# 	batch = buff.getBatch(BATCH_SIZE)
		# 	for j in range(BATCH_SIZE):
		# 		sample = batch[j]
		# 		O[j] = sample[0]
		# 		Next_O[j] = sample[3]
		# 		Matrix[j] = sample[4]
		# 		Next_Matrix[j] = sample[5]

		# 	# para_tr=para_model(torch.Tensor(O).cuda())
		# 	# target_para_tr=para_model_tar(torch.Tensor(Next_O).cuda())
		# 	# q_values = model(torch.Tensor(np.concatenate((O, para_tr.detach().numpy()), axis=O.ndim-1)), torch.Tensor(Matrix))
		# 	q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
		# 	# target_q_values = model_tar(torch.Tensor(np.concatenate((Next_O,target_para_tr.detach().numpy()), axis=Next_O.ndim-1)), torch.Tensor(Next_Matrix)).max(dim = Next_Matrix.ndim-1)[0]
		# 	target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda()).max(dim = Next_Matrix.ndim-1)[0]
		# 	# para_tar_q = target_q_values.clone().detach()
		# 	target_q_values = target_q_values.detach().cpu().numpy()
		# 	expected_q = q_values.clone().detach().cpu().numpy()
			
			
		# 	para_reward=[]
   
		# 	for j in range(BATCH_SIZE):
		# 		sample = batch[j]
		# 		# para_reward.append(sample[6])
		# 		for i in range(NODE_MAX):
		# 			expected_q[j][i][sample[1][i][0]] = sample[2][i] + (1-sample[7])*GAMMA*target_q_values[j][i]
					
		# 	# para_reward=np.array(para_reward)
   
		# 	loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
		# 	# para_loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
		# 	# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).pow(2).mean()
		# 	# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
		# 	# para_loss = - torch.Tensor(para_reward).cuda().pow(2).mean()*model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
		# 	# para_loss = torch.Tensor(para_reward).cuda().pow(2).mean()
		# 	optimizer.zero_grad()
		# 	# para_optim.zero_grad()
		# 	loss.backward()
		# 	# para_loss.backward()
		# 	optimizer.step()
		# 	# para_optim.step()
	
		# # if TimeSteps%100==0:
		# # 	f4.write(str(loss.item())+'\n')
		# # 	f5.write(str(para_loss.item())+'\n')
		# # 	f4.flush()
		# # 	f5.flush()
		# # 	f.write(str(score/3600)+'\n')
		# # 	f.flush()
			
		# if TimeSteps%100 == 0:
		# 	model_tar.load_state_dict(model.state_dict())
			# para_model_tar.load_state_dict(para_model.state_dict())
  ##############################
	# f.write(str(score/3600)+'\n')
	# f.flush()
	# score = 0

	all_time_AVGtrtime=env.allNode_delay/(NODE_MAX*MAX_STEP/3.5)
 
	# if i_episode%1==0:
	# 	# f.write(str(score/3600)+'\n')
	# 	# f2.write(str(all_time_AVGpw)+'\n')
	# 	f3.write(str(all_time_AVGtrtime)+'\n')
	# 	# f.flush()
	# 	# f2.flush()
	# 	f3.flush()
		

	if i_episode%1==0:
		# f4.write(str(loss.item())+'\n')
		# f5.write(str(para_loss.item())+'\n')
		# f4.flush()
		# f5.flush()
		f3.write(str(all_time_AVGtrtime)+'\n')
		f3.flush()
		f.write(str(score/3600)+'\n')
		f.flush()
		score = 0
	
	if i_episode < 10:
		continue
	
	for e in range(N_EPOCH):
		
		batch = buff.getBatch(BATCH_SIZE)
		for j in range(BATCH_SIZE):
			sample = batch[j]
			O[j] = sample[0]
			Next_O[j] = sample[3]
			A_d[j] = sample[1]
			Matrix[j] = sample[4]
			Next_Matrix[j] = sample[5]

		q_select = []
		for batch_id in range(BATCH_SIZE):
			state = O[batch_id]
			topology = Matrix[batch_id]
			ad = A_d[batch_id].astype(int)
			action = np.array(ad)
			action_d = action[: , 0]
			# action_d = torch.Tensor(action_d,dtype=torch.int64).unsqueeze(1).cuda()
			action_d = torch.tensor(action_d,dtype=torch.int64).cuda()
			action_d = action_d.unsqueeze(1).cuda()
			q_temp = model(torch.Tensor([state]).cuda(), torch.Tensor(topology).cuda())[0]
			q_eval = q_temp.gather(dim=1, index=action_d).flatten()
			q_select.append(q_eval)
		
		q_select = torch.stack(q_select)

   
		# para_tr=para_model(torch.Tensor(O).cuda())
		# target_para_tr=para_model_tar(torch.Tensor(Next_O).cuda())
		# q_values = model(torch.Tensor(np.concatenate((O, para_tr.detach().numpy()), axis=O.ndim-1)), torch.Tensor(Matrix))
		q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
		# target_q_values = model_tar(torch.Tensor(np.concatenate((Next_O,target_para_tr.detach().numpy()), axis=Next_O.ndim-1)), torch.Tensor(Next_Matrix)).max(dim = Next_Matrix.ndim-1)[0]
		target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda())
		# target_q_values = target_q_values.max(dim = Next_Matrix.ndim-1)[0]
  
		# para_tar_q = target_q_values.clone().detach()
		with torch.no_grad():
			tar_q_max = []
			target_q_values = np.array(target_q_values.cpu().data)
			for batch_id in range(BATCH_SIZE):
				topo = Next_Matrix[batch_id] - onehot
				q_max_i = []
				for n_id in range(NODE_MAX):
					act_spc = np.nonzero(topo[n_id , :])[0]
					m = target_q_values[batch_id][n_id][0]
					for act in act_spc:
						if target_q_values[batch_id][n_id][act] > m:
							m = target_q_values[batch_id][n_id][act]
					q_max_i.append(m)
				q_max_i = torch.Tensor(np.array(q_max_i)).cuda()
				tar_q_max.append(q_max_i)
			tar_q_max = torch.stack(tar_q_max)
							

		expected_q = np.array(q_select.cpu().data)
  
		for j in range(BATCH_SIZE):
			sample = batch[j]
			# para_reward.append(sample[6])
			for i in range(NODE_MAX):
				# expected_q[j][i] = sample[2][i] + (1-sample[7])*GAMMA*tar_q_max[j][i]	
				expected_q[j][i] = sample[2][i] + GAMMA*tar_q_max[j][i]


		
		# target_q_values = np.array(target_q_values.cpu().data)
		# expected_q = np.array(q_values.cpu().data)
		para_reward=[]


		# for j in range(BATCH_SIZE):
		# 	sample = batch[j]
		# 	# para_reward.append(sample[6])
		# 	for i in range(NODE_MAX):
		# 		expected_q[j][i][sample[1][i][0]] = sample[2][i] + (1-sample[7])*GAMMA*target_q_values[j][i]
				   
		# para_reward=np.array(para_reward)

		# loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
		loss = (q_select - torch.Tensor(expected_q).cuda()).pow(2).mean()
		# para_loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
		# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).pow(2).mean()
		# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
		# para_loss = - torch.Tensor(para_reward).cuda().pow(2).mean()*model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
		# para_loss = torch.Tensor(para_reward).cuda().pow(2).mean()
		optimizer.zero_grad()
		# para_optim.zero_grad()
		loss.backward()
		# para_loss.backward()
		optimizer.step()
		# para_optim.step()
	
	# all_time_AVGtrtime=env.allNode_delay/(NODE_MAX*MAX_STEP)
 
	# if i_episode%1==0:
	# 	# f.write(str(score/3600)+'\n')
	# 	# f2.write(str(all_time_AVGpw)+'\n')
	# 	f3.write(str(all_time_AVGtrtime)+'\n')
	# 	# f.flush()
	# 	# f2.flush()
	# 	f3.flush()
		

	if i_episode%1==0:
		f4.write(str(loss.item()/NODE_MAX)+'\n')
		# f5.write(str(para_loss.item())+'\n')
		f4.flush()
		# f5.flush()
		# f3.write(str(all_time_AVGtrtime)+'\n')
		# f3.flush()
		# f.write(str(score/3600)+'\n')
		# f.flush()
		# score = 0
			
	if i_episode%20 == 0:
		# model_tar.load_state_dict(model.state_dict())
		soft_update(model_tar,model,tau)
 
 
	# for e in range(N_EPOCH):
	# 	batch = buff.getBatch(BATCH_SIZE)
	# 	for j in range(BATCH_SIZE):
	# 		sample = batch[j]
	# 		O[j] = sample[0]
	# 		Next_O[j] = sample[3]
	# 		Matrix[j] = sample[4]
	# 		Next_Matrix[j] = sample[5]

	# 	para_tr=para_model(torch.Tensor(O).cuda())
	# 	target_para_tr=para_model_tar(torch.Tensor(Next_O).cuda())
	# 	# q_values = model(torch.Tensor(np.concatenate((O, para_tr.detach().numpy()), axis=O.ndim-1)), torch.Tensor(Matrix))
	# 	q_values = model(torch.cat((torch.Tensor(O).cuda(), para_tr.detach()), dim=O.ndim-1), torch.Tensor(Matrix).cuda())
	# 	# target_q_values = model_tar(torch.Tensor(np.concatenate((Next_O,target_para_tr.detach().numpy()), axis=Next_O.ndim-1)), torch.Tensor(Next_Matrix)).max(dim = Next_Matrix.ndim-1)[0]
	# 	target_q_values = model_tar(torch.cat((torch.Tensor(Next_O).cuda(), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).max(dim = Next_Matrix.ndim-1)[0]
	# 	para_tar_q = target_q_values.clone().detach()
	# 	target_q_values = target_q_values.detach().cpu().numpy()
	# 	expected_q = q_values.clone().detach().cpu().numpy()
		
	# 	para_reward=[]

	# 	for j in range(BATCH_SIZE):
	# 		sample = batch[j]
	# 		para_reward.append(sample[6])
	# 		for i in range(NODE_MAX):
	# 			expected_q[j][i][sample[1][i][0]] = sample[2][i] + (1-sample[7])*GAMMA*target_q_values[j][i]
				
	# 	para_reward=np.array(para_reward)

	# 	loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
	# 	# para_loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
	# 	# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).pow(2).mean()
	# 	para_loss = - model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
	# 	# para_loss = - torch.Tensor(para_reward).cuda().pow(2).mean()*model_tar(torch.cat((torch.Tensor(Next_O).cuda(), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix).cuda()).pow(2).mean()/para_tar_q.pow(2).mean()/1000
	# 	# para_loss = torch.Tensor(para_reward).cuda().pow(2).mean()
	# 	optimizer.zero_grad()
	# 	para_optim.zero_grad()
	# 	loss.backward()
	# 	para_loss.backward()
	# 	optimizer.step()
	# 	para_optim.step()

	# if i_episode%1==0:
	# 	# f4.write(str(loss.item())+'\n')
	# 	# f5.write(str(para_loss.item())+'\n')
	# 	# f4.flush()
	# 	# f5.flush()
	# 	f.write(str(score/3600)+'\n')
	# 	f.flush()
	# 	score = 0
		
	# if i_episode%20 == 0:
	# 	model_tar.load_state_dict(model.state_dict())
	# 	para_model_tar.load_state_dict(para_model.state_dict())
 
	# all_time_AVGpw=env.allNode_pw/(NODE_MAX*MAX_STEP)
	# all_time_AVGtrtime=env.time_avg/(NODE_MAX*MAX_STEP)
 
	# if i_episode%1==0:
	# 	# f.write(str(score/3600)+'\n')
	# 	f2.write(str(all_time_AVGpw)+'\n')
	# 	f3.write(str(all_time_AVGtrtime)+'\n')
	# 	# f.flush()
	# 	f2.flush()
	# 	f3.flush()
		

	

	# if i_episode < 100:
	# 	continue

	# for e in range(N_EPOCH):
		
	# 	batch = buff.getBatch(BATCH_SIZE)
	# 	for j in range(BATCH_SIZE):
	# 		sample = batch[j]
	# 		O[j] = sample[0]
	# 		Next_O[j] = sample[3]
	# 		Matrix[j] = sample[4]
	# 		Next_Matrix[j] = sample[5]

	# 	para_tr=para_model(torch.Tensor(O))
	# 	target_para_tr=para_model_tar(torch.Tensor(Next_O))
	# 	# q_values = model(torch.Tensor(np.concatenate((O, para_tr.detach().numpy()), axis=O.ndim-1)), torch.Tensor(Matrix))
	# 	q_values = model(torch.cat((torch.Tensor(O), para_tr.detach()), dim=O.ndim-1), torch.Tensor(Matrix))
	# 	# target_q_values = model_tar(torch.Tensor(np.concatenate((Next_O,target_para_tr.detach().numpy()), axis=Next_O.ndim-1)), torch.Tensor(Next_Matrix)).max(dim = Next_Matrix.ndim-1)[0]
	# 	target_q_values = model_tar(torch.cat((torch.Tensor(Next_O), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).max(dim = Next_Matrix.ndim-1)[0]
	# 	target_q_values = target_q_values.detach().numpy()
	# 	expected_q = q_values.clone().detach().numpy()
		
	# 	for j in range(BATCH_SIZE):
	# 		sample = batch[j]
	# 		for i in range(NODE_MAX):
	# 			expected_q[j][i][sample[1][i][0]] = sample[2][i] + GAMMA*target_q_values[j][i]
		
	# 	loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
	# 	# para_loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
	# 	# para_loss = - model_tar(torch.cat((torch.Tensor(Next_O), target_para_tr.detach()), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).pow(2).mean()
	# 	para_loss = - model_tar(torch.cat((torch.Tensor(Next_O), para_tr), dim=Next_O.ndim-1), torch.Tensor(Next_Matrix)).pow(2).mean()/i_episode
	# 	optimizer.zero_grad()
	# 	para_optim.zero_grad()
	# 	loss.backward()
	# 	para_loss.backward()
	# 	optimizer.step()
	# 	para_optim.step()
  
	# 	if i_episode%5==0:
	# 		f4.write(str(loss.item())+'\n')
	# 		f5.write(str(para_loss.item())+'\n')
	# 		f4.flush()
	# 		f5.flush()
   
	# if i_episode%20 == 0:
	# 	model_tar.load_state_dict(model.state_dict())
	# 	para_model_tar.load_state_dict(para_model.state_dict())