from skimage.viewer import ImageViewer
from skimage.transform import rotate
from skimage import transform, color, exposure
from collections import deque
from argparse import ArgumentParser
from torchsummary import summary
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage as skimage
import random
import os

sys.path.append('game/')

import wrapped_flappy_bird as game

sys.path.pop()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------
# parameter

ACTIONS = 2  # number of valid actions
BATCH_SIZE = 32
LR = 1e-3
EPSILON = 0.001
FINAL_EPSILON = 0.0001
OBSERVE = 1000
EXPLORE = 3000000
GAMMA = 0.99
TARGET_REPLACE_ITER = 100
MEMORY_SIZE = 40000
# ----------------------------------------------------------------


class NET(nn.Module):
    def __init__(self, ):
        super(NET, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 256),
            nn.ReLU()
        )
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x);
        x = self.conv2(x);
        x = self.conv3(x);
        x = x.view(x.size(0), -1)
        x = self.fc1(x);
        return self.out(x)


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = NET().to(device), NET().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # ----------------------------------------------------------------
    # choose action
    def choose(self, x):
        x = torch.FloatTensor(x).to(device)  # 轉成tensor
        a_t = np.zeros([ACTIONS])
        # greedy
        if np.random.uniform() > epsilon:
            print("GREEDY",end='      ')
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1]
            a_t[action] = 1
        # random
        else:
            print("RANDOM",end='      ')
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        print(a_t)
        return a_t
    # ----------------------------------------------------------------

    def store(self, s, a, r, s_, ter):
        D.append((s_t, a_t, r_t, s_t1, terminal))
        

        if len(D) > MEMORY_SIZE:
            D.popleft()

        
        self.memory_counter += 1

    def load(self):
        # check flie path
        print(os.getcwd())
        filepath = os.getcwd()+"/save.pt"
        print(filepath)
        time.sleep(0.5)
        # check file exsit
        if os.path.isfile(filepath):
            print("load model param")
            self.eval_net=torch.load('save.pt',map_location=torch.device('cpu'))
            self.target_net=torch.load('save.pt',map_location=torch.device('cpu'))
        else:
            print("no model exist")

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # eval_net parameters to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1
        # sample a minibatch to train
        minibatch = random.sample(D, BATCH_SIZE)
        # unzip minibatch
        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch) 
        
        # reshape action
        action_t = np.array(action_t)
        index = action_t.argmax(axis=1)
        print("action " + str(index))
        index = np.reshape(index, [BATCH_SIZE, 1])
        action_batch_tensor = torch.LongTensor(index).to(device)
        #action = torch.LongTensor(action).to(device)
        state_t = torch.FloatTensor(np.concatenate(state_t)).to(device)
        state_t = state_t.to(device)
        state_t1 = torch.FloatTensor(np.concatenate(state_t1)).to(device)
        state_t1 = state_t1.to(device)
        reward_t = torch.FloatTensor(reward_t).to(device)
        reward_t = torch.reshape(reward_t, (32, 1))
        q_eval = self.eval_net(state_t).gather(1, action_batch_tensor)
        q_eval = torch.reshape(q_eval,(32,1))
        # add detach : dont want update target_net
        q_next = self.target_net(state_t1).detach()
        q_next_reshape = q_next.max(1)[0]
        q_next_reshape = torch.reshape(q_next_reshape, (32, 1))
        q_target = torch.zeros(32, 1).to(device)
        #----------------------------------------------------------------------
        for i in range(BATCH_SIZE):
            if terminal[i]:
                q_target[i] = reward_t[i]
            else:
                q_target[i] = reward_t[i] + GAMMA * q_next_reshape[i]
        #----------------------------------------------------------------------
        loss = self.loss_func(q_eval, q_target)
        print("##############        loss       ##################",loss)

        # back propagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    
    print("#######     device     :",device)
    time.sleep(1)

    
    parser = ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    
    

    dqn = DQN()
    D = deque()
    epsilon = EPSILON
    
    if args['mode'] == 'Run':
        epsilon = 0
        OBSERVE = 999999999999
        dqn.load()
    else :
        dqn.load()

    # record 
    max = 0
    
    for i_episode in range(50000):
        
        # ----------------------------------------------------------------
        # initial environment
        game_state = game.GameState()
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1  # [1,0] do nothing   [0,1] flap
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = skimage.color.rgb2gray(x_t)
        x_t = skimage.transform.resize(x_t, (80, 80))
        x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
        x_t = x_t / 255.0
        s_t = np.stack((x_t, x_t, x_t, x_t))
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
        # ----------------------------------------------------------------
        counter = 0
        pass_count = 0
        while True:
            
            counter += 1
            if counter > max:
                max = counter
            print("######    ",counter,"    ######     ",max,"    ######    ",pass_count,"   #####  ",epsilon)
            # choose action
            a_t = dqn.choose(s_t)
            #print("Episode  ",i_episode,"    choose   ",a_t)
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            if r_t == 1:
                pass_count += 1

            #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON:
                epsilon -= (EPSILON - FINAL_EPSILON) / EXPLORE

            # new env in choose action
            x_t1 = skimage.color.rgb2gray(x_t1_colored)
            x_t1 = skimage.transform.resize(x_t1, (80, 80))
            x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
            x_t1 = x_t1 / 255.0  
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
            s_t1 = np.append(s_t[:, :3, :, :],x_t1, axis=1)  # [1,4,84,84]

            # store the information to queue
            dqn.store(s_t, a_t, r_t, s_t1, terminal)

            # update network
            if dqn.memory_counter > OBSERVE:
                dqn.learn()

            if terminal:
                break
            if dqn.memory_counter % 1000 == 0:
                torch.save(dqn.target_net,'save.pt')
            s_t = s_t1
        
