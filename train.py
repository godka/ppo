import multiprocessing as mp
import numpy as np
import logging
import os
import sys
import gym
#import a2c as network
import ppo as network
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = 4
A_DIM = 2
ACTOR_LR_RATE =1e-4
NUM_AGENTS = 2
TRAIN_SEQ_LEN = 500  # take as a train batch
TRAIN_EPOCH = 2000
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results/ppo'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)

    summary_vars = [td_loss, eps_total_reward]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    with tf.Session() as sess:

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        summary_ops, summary_vars = build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor

        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            total_reward, total_td_loss = [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_, done = exp_queues[i].get()

                v_, td_ = actor.compute_v(s_, a_, g_, done)

                s += s_
                a += a_
                p += p_
                g += v_

                total_reward.append(np.sum(g_))
                total_td_loss.append(np.sum(td_))

            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(g)

            for _ in range(actor.training_epo):
                actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
        
            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: np.mean(total_td_loss),
                summary_vars[1]: np.mean(total_reward)
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()
            
def agent(agent_id, net_params_queue, exp_queue):
    env = gym.make("CartPole-v0")
    env.force_mag = 100.0
    with tf.Session() as sess:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM)))

                action_cumsum = np.cumsum(action_prob)
                action = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                obs, rew, done, info = env.step(action)

                action_vec = np.zeros(A_DIM)
                action_vec[action] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break
            exp_queue.put([s_batch, a_batch, p_batch, r_batch, done])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
