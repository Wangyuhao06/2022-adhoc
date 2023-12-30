import numpy as np
import torch
from parameter import *
#size_normal_list = ((np.random.normal(0,1,parameter.Max_Time)*16+40)//8)*8
#print(size_normal_list)
# sk=[5,6,4]
# q=[1,2,3]
# q=[q,sk]
# print(np.array(q).shape)
# q=np.array([[2,3,5],[1,2,4]])
# a = q.argmax()#.item()
# q=[1]*5

# # for i in bin(10)[2:]:
# p=[[1,2],[3,4]]
# a=np.array([p])
# print(a)
# p=[[5,6],[7,8]]
# b=np.array([p])
# print(np.concatenate((a, b), axis=2))
# p=[]
# p.append(0)
# p[0]=1
# q=p+[2,3,4]
# print(q[len(q)-3:len(q)])

# m=[[[1,2,3],[2,1,5]],[[3,1,2],[5,3,4]]]
# m=np.array(m)
# print(torch.tensor(m).max(dim=2)[0])
# p=[1,2,3]
# p=np.array(p)
# print(np.array([p]))

# print(np.array([[0]*OBS_LEN]*NODE_MAX))

#map()函数映射求两个列表相乘

# func = lambda x,y:x*y
# result = map(func,[1,2,3,4],np.array([4,3,2,1]))
# list_result = list(result)
# list_result=[0,1,0,1,0]
# print(range(list_result))
# print(np.random.randint(10))
# print(abs(np.random.normal(0,1,5)*NOISE_CONST/2))
# ts=torch.tensor([1])
# print(ts.item())
from src.node import Node
from pymobility.models.mobility import random_waypoint
rw = random_waypoint(200, dimensions=(100, 100), velocity=(0.1, 1.0), wt_max=1.0)

