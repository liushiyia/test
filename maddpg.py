# -*- coding: utf-8 -*-
# @Date       : 2024/5/22 19:13
# @Author     : Wang.zr
# @File name  : maddpg.py
# @Description:

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from base.replaybuffer import ReplayBuffer
from base.trainer import ACAgent, Trainer
import numpy as np
import gym #与环境进行交互
from ..common.distribution import gen_action_for_discrete, gen_action_for_continuous #生成连续或离散的动作
from utils.logger import set_logger #记录训练日志

logger = set_logger(__name__, output_file="maddpg.log")
# 在任何使用到的地方
# logger.info("Start training")

DATA_TYPE = tf.float64

"""
构建单个智能体的神经网络（Actor和Critic）
管理目标网络（Target Networks）
提供动作生成、模型保存/加载功能
"""
class MADDPGAgent(ACAgent):
    def __init__(self, name, action_dim, obs_dim, agent_index, args, local_q_func=False):
        super().__init__(name, action_dim, obs_dim, agent_index, args)
        # 初始化网络
        self.name = name + "_agent_" + str(agent_index)
        # self.act_dim = action_dim[agent_index] #动作维度
        # self.obs_dim = obs_dim[agent_index][0] #观测维度
        # 初始化时保存所有智能体的观测和动作维度
        self.obs_dims = [dim[0] for dim in obs_dim]  # 所有智能体的观测维度列表（[leader,follower1,follower2]）
        self.action_dims = action_dim                # 所有智能体的动作维度列表
        # 当前智能体的观测和动作维度
        self.obs_dim = self.obs_dims[agent_index]  # 当前智能体的观测维度（例如 10）
        self.act_dim = self.action_dims[agent_index]  # 当前智能体的动作维度

        self.act_total = sum(action_dim) #所有智能体的动作维度
        self.obs_total = sum([obs_dim[i][0] for i in range(len(obs_dim))]) #总观测维度

        self.num_units = args.num_units #神经网络隐藏层单元数
        self.local_q_func = local_q_func #是否使用局部Q函数（DDPG模式）
        self.nums_agents = len(action_dim) #智能体总数量
        self.actor = self.build_actor() #构建Actor网络
        self.critic = self.build_critic() #构建Critic网络

        self.target_actor = self.build_actor() #构建目标Actor网络
        self.target_critic = self.build_critic() #构建目标Critic网络

        self.actor_optimizer = tf.keras.optimizers.Adam(args.lr) #Actor优化器
        self.critic_optimizer = tf.keras.optimizers.Adam(args.lr) #Critic优化器

    #构建Actor网络（对照作者绘制框图更便于理解）
    def build_actor(self, action_bound=None):
        obs_input = Input(shape=(self.obs_dim,)) #输入层（观测量）
        # TODO FOR DEBUG START 输入层归一化
        normalized = tf.keras.layers.BatchNormalization(axis=-1)(obs_input)
        # TODO FOR DEBUG END 输入层归一化
        out = Dense(self.num_units, activation='relu')(normalized) #隐藏层1（ReLU激活）
        out = Dense(self.num_units, activation='relu')(out) #隐藏层2（ReLU激活）
        out = Dense(self.act_dim * 2, activation=None)(out) #输出层（直接输出动作值）/适用于连续动作空间
        out = tf.cast(out, DATA_TYPE) #确保输出数据类型一致
        actor = Model(inputs=obs_input, outputs=out) #定义Keras模型
        return actor

    # 构建Critic网络
    def build_critic(self):
        if self.local_q_func:  # DDPG模式（仅当前智能体）
            obs_input = Input(shape=(self.obs_dims[self.agent_index],))
            act_input = Input(shape=(self.action_dims[self.agent_index],))
            concatenated = Concatenate(axis=1)([obs_input, act_input])
        if not self.local_q_func:  # MADDPG模式（全局信息）
            #为所有智能体创建输入层，使用各自的观测和动作维度
            obs_input_list = [Input(shape=(dim,)) for dim in self.obs_dims]
            act_input_list = [Input(shape=(dim,)) for dim in self.action_dims]
            concatenated_obs = Concatenate(axis=1)(obs_input_list)
            concatenated_act = Concatenate(axis=1)(act_input_list)
            concatenated = Concatenate(axis=1)([concatenated_obs, concatenated_act])
            # # TODO FOR DEBUG START 输入层归一化
            normalized = tf.keras.layers.BatchNormalization(axis=-1)(concatenated)
            # # TODO FOR DEBUG END 输入层归一化
        out = Dense(self.num_units, activation='relu')(normalized) #隐藏层1
        out = Dense(self.num_units, activation='relu')(out) #隐藏层2
        out = Dense(1, activation=None)(out) #输出值（Q值）
        out = tf.cast(out, DATA_TYPE) #确保输出数据类型一致

        #定义Critic模型的输入（根据模式选择输入）
        critic = Model(inputs=obs_input_list + act_input_list if not self.local_q_func else [obs_input, act_input],
                       outputs=out)

        return critic

    # @tf.function将python函数转换为tensorflow静态图，加速计算
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def agent_action(self, obs):
        return self.actor(obs) # 通过Actor网络生成动作

    @tf.function
    def agent_critic(self, obs_act):
        return self.critic(obs_act) # 通过Critic网络计算Q值

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def agent_target_action(self, obs):
        return self.target_actor(obs) # 目标Actor生成动作

    @tf.function
    def agent_target_critic(self, obs_act):
        return self.target_critic(obs_act) # 目标Critic计算Q值

    # 模型保存与加载
    def save_model(self, path):
        actor_path = f"{path}_{self.name}_actor.h5"
        critic_path = f"{path}_{self.name}_critic.h5"

        self.actor.save(actor_path) # 保存Actor模型
        self.critic.save(critic_path) # 保存Critic模型

        print(f"Actor model saved at {actor_path}")
        print(f"Critic model saved at {critic_path}")

    def load_model(self, path):
        actor_path = f"{path}_{self.name}_actor.h5"
        critic_path = f"{path}_{self.name}_critic.h5"

        self.actor = tf.keras.models.load_model(actor_path) # 加载Actor模型
        self.critic = tf.keras.models.load_model(critic_path) # 加载Critic模型

        print(f"Actor model loaded from {actor_path}")
        print(f"Critic model loaded from {critic_path}")


# 管理训练流程，包括经验回放、网络更新等
class MADDPGTrainer(Trainer):
    def __init__(self, name, obs_dims, action_space, agent_index, args, local_q_func=False):
        super().__init__(name, obs_dims, action_space, agent_index, args, local_q_func)
        self.name = name
        self.args = args
        self.agent_index = agent_index # 当前智能体的索引
        self.nums = len(obs_dims) # 智能体总数量

        # ======================= env preprocess (预处理器)处理动作空间类型（离散或连续）=========================
        self.action_space = action_space
        if isinstance(action_space[0], gym.spaces.Box):
            self.act_dims = [self.action_space[i].shape[0] for i in range(self.nums)]
            self.action_out_func = gen_action_for_continuous # 连续动作生成函数
        elif isinstance(action_space[0], gym.spaces.Discrete):
            self.act_dims = [self.action_space[i].n for i in range(self.nums)]
            self.action_out_func = gen_action_for_discrete # 离散动作生成函数
        # print("动作维度：",self.act_dims)

        # ====================== hyperparameters 超参数=========================
        self.local_q_func = local_q_func
        if self.local_q_func: # 训练算法选用
            logger.info(f"Init {agent_index} is using DDPG algorithm")
        else:
            logger.info(f"Init {agent_index} is using MADDPG algorithm")
        self.gamma = args.gamma #折扣因子
        self.tau = args.tau # 目标网络软更新系数（通常为0.01）
        self.batch_size = args.batch_size # 批次大小

        self.agent = MADDPGAgent(name, self.act_dims, obs_dims, agent_index, args, local_q_func=local_q_func)
        self.replay_buffer = ReplayBuffer(args.buffer_size) # 经验回放缓冲区
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len # 缓冲区最大训练样本数
        self.replay_sample_index = None

        # ====================initialize target networks 初始化目标网络参数（与主网络一致）====================
        self.update_target(self.agent.target_actor.variables, self.agent.actor.variables, tau=self.tau)
        self.update_target(self.agent.target_critic.variables, self.agent.critic.variables, tau=self.tau)

    # 训练
    def train(self, trainers, t):

        if len(self.replay_buffer) <  self.max_replay_buffer_len:  # 检查缓冲区是否足够训练
            return None, None    # 返回 None 表示未训练
        # print("Current replay buffer length:", len(self.replay_buffer))
        # logger.info('Starting training...')

        if not t % 10 == 0:  # 每10步训练一次
            return None, None   # 返回 None 表示未训练
        # print("Current train_step:", t)

        obs_n, action_n, reward_i, next_obs_n, done_i = self.sample_batch_for_pretrain(trainers) # 采样批次数据
        # ======================== train critic 训练Critic网络   ==========================
        with tf.GradientTape() as tape:
            target_actions = [trainer.get_target_action(next_obs_n[i]) for i, trainer in enumerate(trainers)] # 计算目标动作（所有智能体的目标Actor网络生成的下一状态动作）
            #  ============= target ===========
            target_q_input = next_obs_n + target_actions  # 拼接目标Critic网络输入（下一状态和目标动作）
            if self.local_q_func:
                target_q_input = [next_obs_n[self.agent_index], target_actions[self.agent_index]] # DDPG模式下目标Critic网络输入
            target_q = self.agent.agent_target_critic(target_q_input) # 计算目标Q值
            y = reward_i + self.gamma * (1 - done_i) * target_q  # 计算目标y = 当前奖励 + γ * (1 - done) * 目标Q值

            # ============= current 计算目前Q值===========
            q_input = obs_n + action_n  # 当前Critic网络输入（所有智能体状态和动作）
            if self.local_q_func:  # DDPG模式（局部信息）
                q_input = [obs_n[self.agent_index], action_n[self.agent_index]]
            q = self.agent.agent_critic(q_input) #计算当前Critic网络的Q值
            critic_loss = tf.reduce_mean(tf.square(y - q)) # Critic网络损失函数
        critic_grads = tape.gradient(critic_loss, self.agent.critic.trainable_variables) # 梯度计算
        self.agent.critic_optimizer.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables)) # 更新Critic网络

        # ========================= train actor 训练Actor网络===========================
        with tf.GradientTape() as tape:
            _action_n = []
            for i, trainer in enumerate(trainers):
                _action,_,_,_ = trainer.get_action(obs_n[i])
                _action_n.append(_action) # 所有智能体通过Actor网络生成动作
            q_input = obs_n + _action_n # 当前Critic网络输入（所有智能体状态和动作）
            if self.local_q_func:
                q_input = [obs_n[self.agent_index], _action_n[self.agent_index]] # DDPG模式
            p_reg = tf.reduce_mean(tf.square(_action_n[self.agent_index]))  # L2正则化
            q_actor = self.agent.agent_critic(q_input)
            actor_loss = - tf.reduce_mean(self.agent.agent_critic(q_input)) + p_reg * 1e-1 # Actor网络损失函数：最大化Critic的Q值+正则化项
        actor_grads = tape.gradient(actor_loss, self.agent.actor.trainable_variables) # 梯度计算
        self.agent.actor_optimizer.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables)) # 更新Actor网络
        # ======================= update target networks 目标网络更新===================
        self.update_target(self.agent.target_actor.variables, self.agent.actor.variables, self.tau)
        self.update_target(self.agent.target_critic.variables, self.agent.critic.variables, self.tau)

        # TODO FOR DEBUG 保存损失数据，绘制损失函数曲线
        return critic_loss.numpy(), actor_loss.numpy()
        # TODO FOR DEBUG END

        # print("train over")

    def pretrain(self):
        self.replay_sample_index = None

    def save_model(self, path):
        checkpoint = tf.train.Checkpoint(agents=self.agent)
        checkpoint.save(path)

    def load_model(self, path):
        self.agent.load_model(path)

    @tf.function
    def get_action(self, state):
        # pdtype = self.agent.actor(state)
        return self.action_out_func(self.agent.actor(state))  # 返回动作和统计量

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=DATA_TYPE)])
    def get_target_action(self, state):
        action, _, _, _ = self.action_out_func(self.agent.target_actor(state))
        return action

    def update_target(self, target_weights, weights, tau):
        # 软更新公式：target = target * (1 - tau) + main * tau
        for (target, weight) in zip(target_weights, weights):
            target.assign(weight * tau + target * (1 - tau))

    def experience(self, state, action, reward, next_state, done, terminal):
        # 存储经验到缓冲区
        self.replay_buffer.add(state, action, reward, next_state, float(done))

    def sample_batch_for_pretrain(self, trainers):
        # 从所有代理的缓冲区中采样同一批次的索引
        if self.replay_sample_index is None:
            self.replay_sample_index = self.replay_buffer.make_index(self.batch_size)
        obs_n, action_n, next_obs_n = [], [], []
        reward_i, done_i = None, None
        for i, trainer in enumerate(trainers):
            obs, act, rew, next_obs, done = trainer.replay_buffer.sample_index(self.replay_sample_index)
            obs_n.append(obs)
            action_n.append(act)
            next_obs_n.append(next_obs)

            if self.agent_index == i:
                done_i = done
                reward_i = rew
        return obs_n, action_n, reward_i[:, np.newaxis], next_obs_n, done_i[:, np.newaxis]
