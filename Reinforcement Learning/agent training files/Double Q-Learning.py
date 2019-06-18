### ELU 616 
### Author:  NGUYEN Van-Khoa, Pape Samba Diallo this file are based on the reinforcement files written by Carlos Lassance, Nicolas Farrugia
### Improvement: In this file, we used the Double Q learning method introduced by Hando van Hasselt.

### Lab Session on Reinforcement learning
### The goal of this short lab session is to quickly put in practice a simple way to use reinforcement learning to train an agent to play PyRat
### You will be using Experience Replay: while playing, the agent will 'remember' the moves he performs, and will subsequently train itself to predict what should be the next move, depending on how much reward is associated with the moves.

### Usage : python main.py
### Change the parameters directly in this file. 

### GOAL : complete this file (main.py) in order to perform the full training and testing procedure using reinforcement learning and experience replay.

### When training is finished, copy both the AIs/numpy_rl_reload.py file and the save_rl folder into your pyrat folder, and run a pyrat game with the appropriate parameters using the numpy_rl_reload.py as AI

import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from AIs import manh, numpy_rl_reload
import matplotlib.pyplot as plt
import copy

### The game.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent
import game

### The rl.py file describes the reinforcement learning procedure, including Q-learning, Experience replay, and Stochastic Gradient Descent (SGD). SGD is used to approximate the Q-function
import rl

epoch = 10000 ### Total number of epochs that will be done

max_memory = 10000 # Maximum number of experiences we are storing
number_of_batches = 8 # Number of batches per epoch
batch_size = 32 # Number of experiences we use for training per batch
width = 21 # Size of the playing field
height = 15 # Size of the playing field
cheeses = 40 # number of cheeses in the game
opponent = manh # AI used for the opponent

### If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters
load = False
save = True

# After tau steps we update the Q_Target Network
tau0 = 1

# Count variable determines the size of pre-simulation of the games
count0 = 0

env = game.PyRat()
exp_replay = rl.ExperienceReplay(max_memory=max_memory)

# Here we propose the Double Q-learning
model_Q_0 = rl.NLinearModels(env.observe()[0])
model_Target_0 = rl.NLinearModels(env.observe()[0])

if load:
    model_Q_0.load()

# Initiate the Target network
model_Target_0 = copy.deepcopy(model_Q_0)

def predict_action(explore_start, explore_stop, decay_rate, decay_step_, state, model):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step_)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice([0,1,2,3])
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = model.predict([state])
        
        # Take the biggest Q value (= the best action)
        action = np.argmax(Qs)
                
    return action, explore_probability


def play(model_Q,model_Target,tau , count_,epochs,train=True):

    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    loss = 0.
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0
    count = count_
    
    # The probability of exploring the enviroment
    explore_probability = 0.99
    # Decay step deciding how to choose an action for a next state
    # Exploration parameters
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.0001
    decay_step = 0
    t = 0

    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False
        input_t = env.observe()
        t += 1
        
        while not game_over:
            
            decay_step += 1
            
            if count < batch_size:
                count += 1
            else:
                count = batch_size + 100
                
            input_tm1 = input_t          

            # Using the model_Q for predict the target and Use model_Target for evaluate the new Q_value
            #q = model_Q.predict([input_tm1])
            #action = np.argmax(q[0])
            action, explore_probability = predict_action(max_epsilon, min_epsilon, decay, decay_step, input_tm1, model_Q)

            # apply action, get rewards and new state, input_t is a new state
            input_t, reward, game_over = env.act(action)
            if game_over: # Statistics
                steps += env.round # Statistics
                t = 0
                if env.score > env.enemy_score: # Statistics
                    win_cnt += 1 # Statistics
                elif env.score == env.enemy_score: # Statistics
                    draw_cnt += 1 # Statistics
                else: # Statistics
                    lose_cnt += 1 # Statistics
                cheese = env.score # Statistics
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)                
                  
        win_hist.append(win_cnt) # Statistics
        cheeses.append(cheese) # Statistics
        
          
        # Training during the games
        if train:
            local_loss = 0
            for _ in range(number_of_batches):                
                inputs, targets = exp_replay.get_batch(model_Q, model_Target, batch_size=batch_size)
                batch_loss = model_Q.train_on_batch(inputs,targets)
                local_loss += batch_loss
            loss += local_loss
            
        # Update the target network after t steps
        if (t+1) % tau ==0:
            model_Target = copy.deepcopy(model_Q)


        if (e+1) % 100 == 0: # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            string = "Epoch {:03d}/{:03d} | Loss {:.4f} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}| explore_probability {}".format(
                        e,epochs, loss, cheese_np.sum(), 
                        cheese_np[-100:].sum(),win_cnt,draw_cnt,lose_cnt, 
                        win_cnt-last_W,draw_cnt-last_D,lose_cnt-last_L,steps/100, explore_probability)
            print(string)
            loss = 0.
            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt                

print("Training")
play(model_Q_0, model_Target_0,tau0, count0, epoch,True)
if save:
    model_Q_0.save()
print("Training done")
print("Testing")
play(model_Q_0, model_Target_0,tau0, count0, epoch,False)
print("Testing done")
