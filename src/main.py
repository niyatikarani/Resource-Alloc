import multiprocessing
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import numpy as np
from Algorithm import Algorithm
from env import Env
import warnings
warnings.simplefilter("ignore", UserWarning)
#####################  hyper parameters  ####################
CHECK_EPISODE = 4
LEARNING_MAX_EPISODE = 10
MAX_EP_STEPS = 3000
TEXT_RENDER = False
SCREEN_RENDER = True
CHANGE = False
SLEEP_TIME = 0.001

#####################  function  ####################


def exploration(a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ###################################
    
    
    # make directory
if __name__ == "__main__":
    env = Env()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()
   # print(s_dim,"\n", r_dim,"\n", b_dim,"\n", o_dim,"\n", r_bound,"\n", b_bound,"\n", task_inf,"\n", limit, "\n",location)
    algorithm = Algorithm(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    b_var = 1
    ep_reward = []
    r_v, b_v = [], []
    hit_storage = []
    delay_storage = []
    var_reward = []
    reward_storage = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []
    s = env.reset()
    env.initial_screen_demo()
    fig = plt.figure()
    axes = fig.subplots(nrows=2, ncols=2)
    ax1,ax2,ax3,ax4 = axes[0, 0],axes[0, 1],axes[1, 0],axes[1, 1]
    r_plot, = ax1.plot([],[], 'r')
    ax1.set_title('Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    rv_plot, bv_plot, = ax2.plot([],[],[], 'b')
    ax2.set_title('Variance')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Variance')

    h_plot, = ax3.plot([],[], 'g')
    ax3.set_title('HitRate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('HitRate')

    delay, = ax4.plot([],[], 'k')
    ax4.set_title('Delay')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Delay')
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.45, 
                    hspace=0.45)
    plt.show(block=False)

    while var_counter < LEARNING_MAX_EPISODE:
        # initialize
        s = env.reset()
        ep_reward.append(0)

        for j in range(MAX_EP_STEPS):
            # time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()
            env.hit_handler()
            env.delay_handler()
            # choose action according to state
            a = algorithm.choose_action(s)  # a = [R B O]

            # add randomness to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)

            # store the transition parameter
            s_, r = env.algorithm_step_forward(a, r_dim, b_dim)

            algorithm.store_transition(s, a, r / 10, s_)
            # learn
            if algorithm.pointer == algorithm.memory_capacity:
                print("start learning")
            if algorithm.pointer > algorithm.memory_capacity:
                algorithm.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            s = s_
            # sum up the reward
            ep_reward[episode] += r
            # in the end of the episode
            if j == MAX_EP_STEPS - 1:
                var_reward.append(ep_reward[episode])
                reward_storage.append(ep_reward[episode])
                hitlist_avg = env.get_hitlist_avg()
                delay_avg = env.get_delay_avg()
                hit_storage.append(hitlist_avg)
                delay_storage.append(delay_avg)
                env.hitlist_clear()
                env.delay_list_clear()
                r_v.append(r_var)
                b_v.append(b_var)
                print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], '###  r_var: %.2f ' %
                      r_var, 'b_var: %.2f ' % b_var, 'hit_rate: %.2f' % hitlist_avg, 'delay: %.7f' % delay_avg)
                string = 'Episode:%3d' % episode + \
                    ' Reward: %5d' % ep_reward[episode] + '###  r_var: %.2f ' % r_var + \
                    'b_var: %.2f ' % b_var + 'hit_rate: %.2f' % hitlist_avg + 'delay: %.7f' % delay_avg
                env.update_tree(episode, ep_reward[episode], hitlist_avg, delay_avg)
                epoch_inf.append(string)

                r_plot.set_xdata([i+1 for i in range(episode+1)])
                rv_plot.set_xdata([i+1 for i in range(episode+1)])
                bv_plot.set_xdata([i+1 for i in range(episode+1)])
                h_plot.set_xdata([i+1 for i in range(episode+1)])
                delay.set_xdata([i+1 for i in range(episode+1)])

                r_plot.set_ydata(reward_storage)
                rv_plot.set_ydata(r_v)
                bv_plot.set_ydata(b_v)
                h_plot.set_ydata(hit_storage)
                delay.set_ydata(delay_storage)
                ax1.relim()
                ax1.autoscale_view(True,True,True)
                ax2.relim()
                ax2.autoscale_view(True,True,True)
                ax3.relim()
                ax3.autoscale_view(True,True,True)
                ax4.relim()
                ax4.autoscale_view(True,True,True)
                plt.pause(0.01)
                plt.draw()
                # variation change
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1

        # end the episode
        # if SCREEN_RENDER:
        #     env.canvas.tk.destroy()
        episode += 1
   
    dir_name = 'output/' + 'algorithm_'+str(r_dim) + 'u' + str(
        int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location+str(time.time())
    if (os.path.isdir(dir_name)):
        os.rmdir(dir_name)
    os.makedirs(dir_name)
    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i+1 for i in range(episode)], ep_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')
    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')

    fig_hits = plt.figure()
    plt.plot([i + 1 for i in range(episode)], hit_storage)
    plt.xlabel("episode")
    plt.ylabel("hitrate")
    fig_hits.savefig(dir_name + '/hit_rate.png')

    fig_delaytime = plt.figure()
    plt.plot([i + 1 for i in range(episode)], delay_storage)
    plt.xlabel("episode")
    plt.ylabel("delay time in second")
    fig_delaytime.savefig(dir_name + '/delay_time.png')
    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')
    # mean
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE,
          " epochs:", str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" +
            str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    print("the mean of the delay time:" + str(np.mean(delay_storage)))
    f.write("the mean of the delay time:" +
            str(np.mean(delay_storage)) + '\n\n')
    # standard deviation
    print("the standard deviation of the rewards:", str(
        np.std(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" +
            str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    print("the standard deviation of the delay time:", str(np.std(delay_storage)))
    f.write("the standard deviation of the delay time:" +
            str(np.std(delay_storage)) + '\n\n')

    # range
    print("the range of the rewards:", str(
        max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" +
            str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')

    print("the range of the delay time:", str(
        max(delay_storage) - min(delay_storage)))
    f.write("the range of the delay time:" +
            str(max(delay_storage) - min(delay_storage)) + '\n\n')

    f.close()
