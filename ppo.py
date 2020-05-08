import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import gym #OpenAI Gym
import retro #Gym Retro
import numpy as np #NumPy
from time import time #calculate runtime
#TensorFlow 2.0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() #Disable Eager, IMPORTANT!

from wrappers import * #Gym Retro wrappers
from utils import convert_frames, Now, get_latest_file

class PPOAgent:
    def __init__(self, game, combos, time_limit=None, epochs=10, episodes_per_batch=4, minibatch_size=32,
                 alpha=1e-4, beta=1e-4, gamma=0.99, loss_clipping=0.20, entropy_beta=0.005):
        #retro environment
        self.game = game #game rom name
        self.combos = combos #valid discrete button combinations
        self.env = self.build_env(time_limit=time_limit) #retro environment
        self.num_actions = len(combos) #number of possible actions for env
        self.state_shape = self.env.observation_space.shape #env state dims
        self.state = self.reset() #initialize state
        #training
        self.epochs = epochs #number of epochs to fit on
        self.episodes_per_batch = episodes_per_batch #the number of epsiodes to run before fit
        self.minibatch_size = minibatch_size #batch size to fit on
        self.alpha = alpha #learning rate for actor
        self.beta = beta #learning rate for actor
        self.gamma = gamma #reward discount factor
        self.loss_clipping = loss_clipping #0.2 loss clipping recommended by ppo paper
        self.entropy_beta = entropy_beta #0.005 entropy beta recommended by ppo paper
        #memory
        self.states = [] #stores states
        self.actions = [] #stores actions (one hot)
        self.predictions = [] #stores predictions
        self.rewards = [] #stores rewards
        self.masks = [] #stores terminal mask
        self.log = [] #stores information from training
        #models
        self.actor, self.critic, self.policy = self.build_ac_network()

    def build_env(self, time_limit=None, downsampleRatio=2, numStack=4):
    #Build the gym retro environment.
        env = retro.make(game=self.game, state=retro.State.DEFAULT, scenario='scenario',
                         record=False, obs_type=retro.Observations.IMAGE)
        env = Discretizer(env, combos=self.combos)
        if time_limit is not None: env = TimeLimit(env, time_limit)
        env = SkipFrames(env)
        env = Rgb2Gray(env)
        env = Downsample(env, downsampleRatio)
        env = FrameStack(env, numStack)
        env = ScaledFloatFrame(env)
        return env
        
    def build_ac_network(self):
    #Build the actor and critic networks and define the custom loss function.
        # ppo loss algorithm taken from ppo paper https://arxiv.org/abs/1707.06347
        def ppo_loss(old_pred, advantages):
            def loss(y_true, y_pred):
                new_policy = K.sum(y_pred * y_true, axis=-1)
                old_policy = K.sum(old_pred * y_true, axis=-1)
                r = new_policy / (old_policy + 1e-10)
                p1 = r * advantages
                p2 = K.clip(r, 1 - self.loss_clipping, 1 + self.loss_clipping) * advantages
                actor_loss = -K.minimum(p1, p2)
                entropy = self.entropy_beta * -new_policy * K.log(new_policy + 1e-10)
                return actor_loss + entropy
            return loss
        #Input to actor, critic, and policy
        X_input = Input(self.state_shape, name='input')
        #build actor
        old_pred = Input((self.num_actions), name='old_pred') #need to feed in so ppo_loss can access
        advantages = Input(1, name='advantages') #need to feed in so ppo_loss can access
        A = Conv2D(32, 8, 4, activation='relu')(X_input)
        A = Conv2D(64, 4, 2, activation='relu')(A)
        A = Conv2D(64, 3, 1, activation='relu')(A)
        A = Flatten()(A)
        A = Dense(1024, activation='relu', kernel_initializer='he_uniform')(A)
        A = Dense(512, activation='relu', kernel_initializer='he_uniform')(A)
        action = Dense(self.num_actions, activation='softmax', name='action')(A)
        actor = Model(inputs=[X_input, old_pred, advantages], outputs=[action], name='actor')
        actor.compile(loss=ppo_loss(old_pred, advantages), optimizer=Adam(lr=self.alpha))
        #build policy - actor model without extra inputs used for prediction
        policy = Model(inputs=[X_input], outputs=[action], name='policy') 
        #build critic
        C = Conv2D(32, 8, 4, activation='relu')(X_input)
        C = Conv2D(64, 4, 2, activation='relu')(C)
        C = Conv2D(64, 3, 1, activation='relu')(C)
        C = Flatten()(C)
        C = Dense(1024, activation='relu', kernel_initializer='he_uniform')(C)
        C = Dense(512, activation='relu', kernel_initializer='he_uniform')(C)
        value = Dense(1, activation='linear', name='value')(C)
        critic = Model(inputs=[X_input], outputs=[value], name='critic')
        critic.compile(loss='mse', optimizer=Adam(lr=self.beta))
        return actor, critic, policy

    def remember(self, next_state, action, pred, reward, mask):
        #Store data in memory.
        self.states.append(self.state)
        action_onehot = np.zeros([self.num_actions])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.predictions.append(pred)
        self.rewards.append(reward)
        self.masks.append(mask)
        #Update current State
        self.state = next_state
        
    def forget(self):
    #Clear data from memory.
        self.states = []
        self.actions = []
        self.predictions = []
        self.rewards = []
        self.masks = []

    def choose_action(self, training):
    #Predict next action based on current state.
        pred = self.policy.predict(self.state)[0]
        if training: #when training sample action distribution
            action = np.random.choice(self.num_actions, p=pred)
        else: #when playing always choose best action
            action = np.argmax(pred)
        return action, pred

    def discount_rewards(self):
    #Calculate the discounted the rewards.
        rsum = 0.0
        dr = np.zeros_like(self.rewards, dtype=np.float)
        for i in reversed(range(len(self.rewards))):
            rsum = rsum * self.gamma * self.masks[i] + self.rewards[i]
            dr[i] = rsum
        return dr
                
    def learn(self):
    #Train the actor and critic networks.
        #convert to valid shape for training (1, height, width, channels)
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        old_pred = np.vstack(self.predictions)
        #discount rewards and use to calculate advantage
        discounted_rewards = self.discount_rewards()
        values = self.critic.predict(states)[:, 0]
        advantages = discounted_rewards - values
        #train the models for num_epochs epochs and minibatch size < episodes_per_batch * steps
        self.actor.fit([states, old_pred, advantages], [actions], epochs=self.epochs, verbose=0,
                       shuffle=True, batch_size=self.minibatch_size)
        self.critic.fit([states], [discounted_rewards], epochs=self.epochs, verbose=0,
                        shuffle=True, batch_size=self.minibatch_size)
        self.forget() #reset memory
    
    def load(self, directory, actor_name=None, critic_name=None):
    #Load the actor and critic weights.
        print('Loading models ...', end=' ')
        #if no names supplied try to load most recent
        if actor_name is not None and critic_name is not None:
            actor_path = os.path.join(directory, actor_name)
            critic_path = os.path.join(directory, critic_name)
        elif actor_name is None and critic_name is None:
            actor_path = get_latest_file(directory + '/*Actor.h5')
            critic_path = get_latest_file(directory + '/*Critic.h5')
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        print('Done. Models loaded from {}'.format(directory))
        print('Loaded Actor model {}'.format(actor_path))
        print('Loaded Critic model {}'.format(critic_path))

    def save(self, directory, fileName):
    #Save the actor and critic weights.
        print('Saving models ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        actor_name = fileName + '_Actor.h5'
        critic_name = fileName + '_Critic.h5'
        self.actor.save_weights(os.path.join(directory, actor_name))
        self.critic.save_weights(os.path.join(directory, critic_name))
        print('Done. Saved to {}'.format(os.path.abspath(directory)))
        
    def save_log(self, directory, fileName, clear=False):
    #Save the information currently stored in log list.
        print('Saving log ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        f = open(os.path.join(directory, fileName + '.csv'), 'w')
        for line in self.log:
            f.write(str(line)[1:-1].replace('None', '') + '\n')
        f.close()
        if clear: self.log = []
        print('Done. Saved to {}'.format(os.path.abspath(directory)))

    def reset(self):
    #Reset environment and return expanded state.
        self.state = np.expand_dims(self.env.reset(), axis=0)

    def close(self):
    #Close the environment.
        self.env.close()

    def step(self, action):
    #Run one step for given action and return data.
        observation, reward, done, info = self.env.step(action)
        observation = np.expand_dims(observation, axis=0)
        return observation, reward, done, info
    
    def run(self, num_episodes=100, render=False, checkpoint=False, cp_render=False, cp_interval=None, otype='AVI'):
    #Run num_episodes number of episodes and train the actor and critic models after episodes_per_batch
    #number of episodes. If render is true then render each episode to monitor. If checkpoint is true
    #then save model weights and log and evaluate the model and convert and save the frames every
    #cp_interval number of of episodes. The evaluation is rendered if cp_render is true.
        printSTR = 'Episode: {}/{} | Score: {:.4f} | AVG 50: {:.4f} | Elapsed Time: {} mins'
        start_time = time()
        scores = []
        self.reset()
        for e in range(1, num_episodes + 1):
            score = 0
            LIVES = None #will store starting lives
            while True:
                action, pred = self.choose_action(training=True) #sample from action distribution
                next_state, reward, done, info = self.step(action) #perform action
                score += reward #cumulative score for episode
                reward = np.clip(reward, -1.0, 1.0).item() #clip reward to range [-1.0, 1.0]
                if LIVES is None: LIVES = info['lives'] #get starting lives
                if info['lives'] < LIVES: done = True #flag for reset when dead
                self.remember(next_state, action, pred, reward, not done) #store results
                if render: self.env.render()
                if done:
                    scores.append(score) #store scores for all epsisodes
                    self.reset()
                    #when episodes_per_batch number of episodes have ran or if last epsisode then learn
                    if (e % self.episodes_per_batch) == 0 or e == num_episodes: self.learn()
                    break    
            elapsed_time = round((time() - start_time)/60, 2)
            print(printSTR.format(e, num_episodes, round(score, 4), np.average(scores[-50:]), elapsed_time))
            if checkpoint and (e % cp_interval) == 0:
                eval_score, frames = self.evaluate(render=cp_render)
                print('EVALUATION: {}'.format(round(eval_score, 4)))
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, eval_score])
                fileName = 'PPO_{}_{}_{}'.format(e, self.game, Now(separate=False))
                self.save('models', fileName)
                self.save_log('logs', fileName)
                convert_frames(frames, 'renders', fileName, otype=otype)
            elif checkpoint:
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, None])
        elapsed_time = round((time() - start_time)/60, 2)
        print('Finished training {} episodes in {} minutes.'.format(num_episodes, elapsed_time))

    def evaluate(self, render=False):
    #Run an episode and return the score and frames.
        frames = []
        score = 0
        self.reset()
        while True:
            action, _ = self.choose_action(training=False) #get best action
            observation, reward, done, info = self.step(action) #perform action
            self.state = observation #update current state
            score += reward #cumulative score for episode
            if render: self.env.render()
            frames.append(self.env.render(mode='rgb_array'))
            if done:
                self.reset()
                break
        return score, frames

    def play_episode(self, render=False, render_and_save=False, otype='AVI'):
    #Run one episode. If render is true then render each episode to monitor.
    #If render_and_save is true then save frames and convert to GIF image or AVI movie.
    #The reward for the episode is returned.
        frames = []
        score = 0
        self.reset()
        while True:
            action, _ = self.choose_action(training=False) #get best action
            observation, reward, done, info = self.step(action) #perform action
            self.state = observation #update cuurent state
            score += reward #cumulative score for episode
            if render: self.env.render()
            if render_and_save: frames.append(self.env.render(mode='rgb_array'))
            if done:
                print('Finished! Score: {}'.format(score))
                self.reset()
                break
        if render_and_save:
            fileName = 'PPO_PLAY_{}_{}'.format(self.game, Now(separate=False))
            convert_frames(frames, 'renders', fileName, otype=otype)
        return score