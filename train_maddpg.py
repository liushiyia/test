import numpy as np
import time
import pickle
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #屏蔽 INFO 和 WARNING 日志，只输出 ERROR 和 FATAL 日志
import tensorflow as tf

from maddpg.trainer.maddpg import MADDPGTrainer
from base.trainer import MultiTrainerContainer
from base.args_config import parse_args_maddpg
from utils.utilss import load_data2_plot
from utils.logger import set_logger

logger = set_logger(__name__, output_file="train_maddpg.log")

#检查运行
if tf.config.list_physical_devices():
    print("使用GPU运行")
else:
    print("使用CPU运行")

# 在任何使用到的地方
logger.info("Start training")
#-------------------创建多智能体环境----------------------#
def make_env(scenario_name, arglist, benchmark=False):

    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario() #从脚本中加载场景
    world = scenario.make_world() #创建对应的世界

    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.done)
    return env

#------------------------------- 创建所有智能体的训练器-------------------------------------#
def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    trainer = MADDPGTrainer #MADDPG训练器

    # 创建对抗性智能体的训练器
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    # 创建友好性智能体的训练器
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers

#---------------------------------主训练函数----------------------------------#
def train(arglist):
    # ----------------1.Create environment-----------------#
    logger.info("=====================================================================================")
    curtime = datetime.datetime.now()
    cur_dir = f"{curtime.strftime('%Y-%m-%d-%H-%M-%S')}"
    logger.info(f"Training start at {cur_dir}")
    arglist.save_model_dir = arglist.save_model_dir + arglist.exp_name + '/' + arglist.scenario + '/' + cur_dir
    logger.info(f"Save dir: {arglist.save_model_dir}")
    if not os.path.exists(arglist.save_model_dir): # 是否加载已经保存的模型
        os.makedirs(arglist.save_model_dir)

    env = make_env(arglist.scenario, arglist, arglist.benchmark) # 调用智能体环境创建函数

    # ---------------2.Create agent trainers----------------#
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries) # 对抗者数量

    trainers =   get_trainers(env, num_adversaries, obs_shape_n, arglist) #调用训练器（maddpg or ddpg）
    logger.info('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

    #------------------3.checkpoint加载之前模型-----------------#
    # 创建MultiAgentContainer对象
    multi_agent_container = MultiTrainerContainer(trainers)
    checkpoint = tf.train.Checkpoint(multi_agent_container=multi_agent_container)
    # Load previous results, if necessary
    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_model_dir
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, arglist.load_dir, max_to_keep=5)
    if arglist.display or arglist.restore or arglist.benchmark:
        logger.info('Loading previous state...')
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    #----------------------4.变量初始化------------------------#
    episode_rewards = [0.0]  # sum of rewards for all agents in one episode
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve in last episode
    final_ep_ag_rewards = []  # agent rewards for training curve in last episode
    agent_info = [[[]]]  # placeholder for benchmarking info

    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    # TODO FOR plot trajectory start1:
    # Initialize trajectory storage智能体轨迹
    agent_trajectories = {f'agent_{i}': [] for i in range(env.n)}
    agent_vel= {f'agent_{i}': [] for i in range(env.n)}
    all_episodes_trajectories = []
    all_episodes_velocities = []
    # TODO FOR plot trajectory end

    # Initialize formation_error
    tracking_errors = []  # 初始化编队跟踪误差
    # Initialize obs_dist
    obstacle_dists = []  # 初始化与障碍物距离

    # 奖励分量
    agent_reward_components = {f'agent_{i}': {
    'track': [],
    'collision': [],
    'formation': []
    } for i in range(env.n)}

    # 速度记录
    agent_velocities = {f'agent_{i}': [] for i in range(env.n)}
    current_episode_velocities = {f'agent_{i}': [] for i in range(env.n)}

    # 损失函数记录容器
    critic_losses = [[] for _ in range(env.n)] # 每个智能体的critic损失
    actor_losses = [[] for _ in range(env.n)]  # 每个智能体的actor损失

    # 初始化速度统计量
    vel_statistics = {f'agent_{i}': {'mean_x': [], 'mean_y': [], 'logstd_x': [], 'logstd_y': [], 'std_x': [], 'std_y': []}
                      for i in range(env.n)}
    #--------------------5.开始迭代循环训练-------------------#
    logger.info('Starting iterations...')
    while True:
        # get action
        # 动作获取代码形式一：
        # action_n = [trainer.get_action(np.expand_dims(obs, axis=0))[0] for trainer, obs in zip(trainers, obs_n)]
        # 动作获取代码形式二：
        action_n = []
        ##==================0 动作值获取=========================#
        # for trainer, obs in zip(trainers, obs_n):
        #     obs_expanded = np.expand_dims(obs, axis=0) #变成（1,N）的二维输入
        #
        #     # TODO FOR DEBUG 启用eager执行模式强制以python方式运行函数
        #     tf.config.run_functions_eagerly(True)
        #     action = trainer.get_action(obs_expanded) #返回形如[[a1,a2]]的二维数组
        #     tf.config.run_functions_eagerly(False)
        #
        #     #action = trainer.get_action(obs_expanded)  # 返回形如[[a1,a2]]的二维数组
        #     action_n.append(action[0]) #取出第一个动作向量，加入列表
        ##==================1 动作值获取=========================#
        # TODO FOR DEBUG 提取动作速度的统计量（均值、方差）
        for i,(trainer,obs) in enumerate(zip(trainers, obs_n)):
            obs_expanded = np.expand_dims(obs, axis=0)  # 变成（1,N）的二维输入
            tf.config.run_functions_eagerly(True)  # 启用eager执行模式强制以python方式运行函数
            action, mean, logstd, std = trainer.get_action(obs_expanded)
            action_n.append(action[0])
            tf.config.run_functions_eagerly(False)

            # 统计量保存
            agent_id = f'agent_{i}'
            mean = mean[0].numpy()
            logstd = logstd[0].numpy()
            std = std[0].numpy()

            vel_statistics[agent_id]['mean_x'].append(mean[0])
            vel_statistics[agent_id]['mean_y'].append(mean[1])
            vel_statistics[agent_id]['logstd_x'].append(logstd[0])
            vel_statistics[agent_id]['logstd_y'].append(logstd[1])
            vel_statistics[agent_id]['std_x'].append(std[0])
            vel_statistics[agent_id]['std_y'].append(std[1])

        # TODO for debug 记录智能体速度变化
        for i,action in enumerate(action_n):
            action_array = action.numpy()
            agent_id = f'agent_{i}'
            v_x = action_array[0]
            v_y = action_array[1]
            current_episode_velocities[agent_id].append((v_x, v_y))

        # # TODO for debug 最后一幕
        # if (len(episode_rewards[:-1]) > 1) and (len(episode_rewards) % arglist.num_episodes) == 0:
        #     a = 1
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        # 获取奖励分量并记录
        for i, ag in enumerate(env.agents):
            comp = env.world.scenario.get_reward_components(ag, env.world)
            agent_reward_components[f'agent_{i}']['track'].append(comp['track'])
            agent_reward_components[f'agent_{i}']['collision'].append(comp['collision'])
            agent_reward_components[f'agent_{i}']['formation'].append(comp['formation'])
        episode_step += 1
        # TODO for debug 验证障碍物地标是否移动
        # if episode_step % 10 == 0:
        #     for landmark in env.world.landmarks:
        #         print(f"{landmark.name} position: {landmark.state.p_pos}")
        done = any(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        #------------------实时记录轨迹、编队跟踪误差、障碍物距离-----------#
        # 1.Record current positions for trajectory 保存所有智能体所有episodes的轨迹
        # TODO FOR plot trajectory start2:
        current_positions = [agent.state.p_pos for agent in env.agents] #保存智能体当前位置信息
        current_velocities = [agent.state.p_vel for agent in env.agents]
        # print(type(env.agents[0].state.p_pos))  # 应该显示 <class 'numpy.ndarray'>
        for i in range(env.n):
            agent_trajectories[f'agent_{i}'].append(current_positions[i].numpy().tolist())
            agent_vel[f'agent_{i}'].append(current_velocities[i].numpy().tolist())
        # TODO FOR plot trajectory end
        # 2.collect formation_errors
        current_tracking_errors = []
        for agent in env.agents[1:]: #只考虑跟随者
            # 调用 benchmark_data 获取数据
            dist_data = env.world.scenario.benchmark_data(agent, env.world)
            # 最后一个元素是与领航者的距离
            error = abs(dist_data[-1] - 0.4)
            current_tracking_errors.append(error)
        tracking_errors.append(current_tracking_errors)
        # 3.collect obs_dist
        min_obstacle_dist = None #先用领航者与障碍物最小距离对其赋值
        for i, agent in enumerate(env.agents):
            if i == 0:  # 领航者与障碍物的最短距离
                # 领航者直接计算与所有障碍物地标的距离
                obstacle_dists_agent = []
                for landmark in env.world.landmarks[1:]:  # landmarks[0]是目标
                    dist_sq = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                    obstacle_dists_agent.append(dist_sq)
                if obstacle_dists_agent:
                    current_min = min(obstacle_dists_agent)
                    if min_obstacle_dist is None or current_min < min_obstacle_dist:
                        min_obstacle_dist = current_min
            else:  # 跟随者与障碍物的最短距离
                dists = env.world.scenario.benchmark_data(agent, env.world)
                obstacle_dists_agent = dists[:-1]  # 排除与领航者的距离
                if obstacle_dists_agent:  # 确保有障碍物距离
                    current_min = min(obstacle_dists_agent)
                    if min_obstacle_dist is None or current_min < min_obstacle_dist:
                        min_obstacle_dist = current_min
        if min_obstacle_dist is not None:
            obstacle_dists.append(min_obstacle_dist)

        #-------------------实时更新奖励数据------------------#
        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        if done or terminal:
            # TODO FOR plot trajectory start3:
            episode_data = {
                'episode':len(episode_rewards),
                'agent_trajectories': {k: v[:] for k, v in agent_trajectories.items()},
                'landmarks': [landmark.state.p_pos for landmark in env.world.landmarks]
            }
            all_episodes_trajectories.append(episode_data)

            # 新增：保存速度数据
            velocity_episode_data = {
                'episode': len(episode_rewards),
                'agent_vel': {k: v[:] for k, v in agent_vel.items()}
            }
            all_episodes_velocities.append(velocity_episode_data)

            for agent_id in agent_trajectories:
                agent_trajectories[agent_id] = []
                agent_vel[agent_id] = []  # 新增：重置速度数据
            # TODO FOR plot trajectory end

            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])
            # TODO for debug 速度值保存
            for agent_id in agent_velocities.keys():
                agent_velocities[agent_id].append(current_episode_velocities[agent_id][:])
                current_episode_velocities[agent_id].append([])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = os.path.join(arglist.benchmark_dir, arglist.exp_name + '.pkl')
                os.makedirs(os.path.dirname(file_name), exist_ok=True)  # 创建目录
                logger.info('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.2)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        # loss = None
        for agent in trainers:
            agent.pretrain() #从每个智能体各自的经验池中采样（预处理器）
        for i, agent in enumerate(trainers):
            c_loss, a_loss = agent.train(trainers, train_step)  # 训练
            if c_loss is not None:
                critic_losses[i].append(c_loss)
                actor_losses[i].append(a_loss)

        # save model, display training output
        # valid_rewards = episode_rewards[:-1]  # 去除最后一个元素
        if terminal and (len(episode_rewards[:-1]) % arglist.save_rate == 0):
            checkpoint_manager.save()
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                logger.info(
                    "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards[:-1]), np.mean(episode_rewards[-arglist.save_rate-1:-1]),
                        [np.mean(rew[-arglist.save_rate-1:-1]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            else:
                logger.info(
                    "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(   
                        train_step, len(episode_rewards[:-1]), np.mean(episode_rewards[-arglist.save_rate-1:-1]),
                        [np.mean(rew[-arglist.save_rate-1:-1]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            t_start = time.time()
            # Keep track of final episode reward
            final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate-1:-1]))
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate-1:-1]))

        # saves final episode reward for plotting training curve later
        if len(episode_rewards[:-1]) > arglist.num_episodes:
            file_dir = arglist.plots_dir + cur_dir + '/'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            rew_file_name = file_dir + arglist.exp_name + '_rewards.pkl'
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(final_ep_rewards, fp)
            agrew_file_name = file_dir + arglist.exp_name + '_agrewards.pkl'
            with open(agrew_file_name, 'wb') as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            # 保存奖励分量数据
            comp_file = os.path.join(file_dir, arglist.exp_name + '_reward_components.pkl')
            with open(comp_file, 'wb') as fp:
                pickle.dump(agent_reward_components, fp)
            # 保存critic损失变量
            critic_loss_file = os.path.join(file_dir, arglist.exp_name + f"_critic_loss_{cur_dir}.pkl")
            with open(critic_loss_file, 'wb') as fp:
                pickle.dump(critic_losses, fp)
            # 保存actor损失变量
            actor_loss_file = os.path.join(file_dir, arglist.exp_name + f"_actor_loss_{cur_dir}.pkl")
            with open(actor_loss_file, 'wb') as fp:
                pickle.dump(actor_losses, fp)
            # 保存速度变量
            velocities_file = os.path.join(file_dir, arglist.exp_name + '_velocities.pkl')
            with open(velocities_file, 'wb') as fp:
                pickle.dump(agent_velocities, fp)
            # 保存动作速度统计量
            vel_statistics_file = os.path.join(file_dir, arglist.exp_name + f"_statistics.pkl")
            with open(vel_statistics_file, 'wb') as fp:
                pickle.dump(vel_statistics, fp)
            #保存 agents运动轨迹记录
            # TODO FOR plot trajectory start4:
            traj_file = os.path.join(file_dir, arglist.exp_name + '_trajectories.pkl')
            with open(traj_file, 'wb') as fp:
                pickle.dump(all_episodes_trajectories, fp)

            # 保存速度数据
            velocity_file = os.path.join(file_dir, arglist.exp_name + '_velocity.pkl')
            with open(velocity_file, 'wb') as fp:
                pickle.dump(all_episodes_velocities, fp)
            # TODO FOR plot trajectory end

            # 保存 agents与障碍物距离变化
            obstacle_dist_file = os.path.join(file_dir, arglist.exp_name + '_obstacle_dists.pkl')
            with open(obstacle_dist_file, 'wb') as fp:
                pickle.dump(obstacle_dists, fp)
            # 保存编队跟踪误差
            tracking_error_file = os.path.join(file_dir + arglist.exp_name + '_tracking_errors.pkl')
            with open(tracking_error_file, 'wb') as fp:
                pickle.dump(tracking_errors, fp)

            logger.info('...Finished total of {} episodes.'.format(len(episode_rewards[:-1])))

            if arglist.show_plots:
                load_data2_plot(arglist,rew_file_name, "reward", False) #平均幕奖励（每一幕所有智能体奖励之和）
                load_data2_plot(arglist,agrew_file_name, "agreward", False) #智能体平均幕奖励（每一幕中每个智能体奖励）
                # load_data2_plot(arglist,traj_file, "trajectories", False)
                # load_data2_plot(arglist,obstacle_dist_file, "obs_dists", False)
                # load_data2_plot(arglist,tracking_error_file, "formation_error", False)
                # load_data2_plot(arglist, comp_file, "reward_components", False)
            break


if __name__ == '__main__':
    arglist = parse_args_maddpg()
    train(arglist)
