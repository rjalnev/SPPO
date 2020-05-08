from ppo import PPOAgent
#from utils import install_roms_in_folder

GAMES = ['SpaceInvaders-Nes', 'Joust-Nes', 'SuperMarioBros-Nes', 'MsPacMan-Nes']
COMBOS = [ [['LEFT'], ['RIGHT'], ['A'], ['LEFT', 'A'], ['RIGHT', 'A']],
           [['LEFT'], ['RIGHT'], ['UP'], ['DOWN']] ]

if __name__ == '__main__':

    #install roms
    #install_roms_in_folder('roms/')

    #create agent
    ppo = PPOAgent(GAMES[0], COMBOS[0], episodes_per_batch=8, alpha=1e-5, beta=1e-5, entropy_beta=0.01)
    ppo.actor.summary()
    ppo.critic.summary()
    
    #train agent
    ppo.run(num_episodes=10, render=True, checkpoint=False, cp_interval=100, cp_render=True)
    
    #load model
    ppo.load('models', 'PPO_6900_SpaceInvaders_Actor.h5', 'PPO_6900_SpaceInvaders_Critic.h5')
    
    #play game
    ppo.play_episode(render=True, render_and_save=False, otype='AVI')