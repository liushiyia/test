# -*- coding: utf-8 -*-
# @Date       : 2025/4/11
# @Author     : Zhang.yw
# @File name  : utilss.py
# @Description:
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors

from multiagent.scenarios.formation import Scenario


def save_model(arglist, exp_time, times, chechpoint):
    save_exp_dir = arglist.save_model_dir + arglist.exp_name + '/' + exp_time + '/'
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    save_cur_dir = save_exp_dir + str(times) + '/'
    if not os.path.exists(save_cur_dir):
        os.makedirs(save_cur_dir)
    chechpoint.save(save_cur_dir)
    print(f"--Save to {save_cur_dir}")


def load_model(path, chechpoint):
    # 加载模型
    chechpoint.restore(tf.train.latest_checkpoint(path)).expect_partial()

def load_data2_plot(arglist,data_path, name: str, show=True):
    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    defined_episodes = arglist.num_episodes
    steps_per_episode = arglist.max_episode_len

    # 创建基础目录路径
    file_dir = os.path.dirname(data_path) + '/'

    if name == "trajectories":
        plt.figure()
        # 处理字典类型的轨迹数据
        for agent_id, trajectory in data_list.items():
            if isinstance(trajectory[0], list):  # 检查是否为坐标列表
                x = [pos[0] for pos in trajectory]
                y = [pos[1] for pos in trajectory]
                plt.plot(x, y, label=agent_id)
                plt.scatter(x[0], y[0], marker='o')  # 起点
                plt.scatter(x[-1], y[-1], marker='x')  # 终点
        plt.title("Agent Trajectories")
        plt.legend()

        # 从 Scenario 中提取障碍和目标点
        scenario = Scenario()
        world = scenario.make_world()
        goal_pos = world.landmarks[0].state.p_pos
        for i, lm in enumerate(world.landmarks[1:]):
            obs_pos = lm.state.p_pos
            plt.scatter(obs_pos[0], obs_pos[1], color='black', marker='X', s=100, label='Obstacle' if i == 0 else "")
        plt.scatter(goal_pos[0], goal_pos[1], color='red', marker='*', s=150, label='Goal')

    elif name == "reward":  # 平均幕奖励（所有智能体奖励之和）
        plt.figure()
        truncated_data = data_list[:defined_episodes]
        plt.plot(truncated_data)
        plt.title("Episode Total Reward (Sum of Agents)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.xlim(0, defined_episodes - 1)

    elif name == "agreward":  # 每个智能体的平均幕奖励
        plt.figure()
        truncated_data = data_list[:defined_episodes * 3]  # 每幕3个智能体

        # 将数据拆分为三个智能体的奖励序列
        num_agents = 3
        agent_rewards = [[] for _ in range(num_agents)]

        # 每3个数据为一组（三个智能体）
        for i in range(0, len(truncated_data), num_agents):
            for j in range(num_agents):
                if i + j < len(truncated_data):
                    agent_rewards[j].append(truncated_data[i + j])

        # 绘制每个智能体的奖励曲线
        for idx, rewards in enumerate(agent_rewards):
            plt.plot(rewards, label=f'Agent {idx}')
        plt.title("Agent Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.xlim(0, defined_episodes - 1)

    elif name == "obs_dists": # 智能体与障碍物最短距离
        plt.figure()
        truncated_data = data_list[:defined_episodes * steps_per_episode]

        # 提取最后一幕的数据
        last_episode_start = (defined_episodes - 1) * steps_per_episode
        last_episode_data = truncated_data[last_episode_start:last_episode_start + steps_per_episode]

        plt.plot(last_episode_data)
        plt.title("Minimum Distance to Obstacles (Last Episode)")
        plt.xlabel("Step in Last Episode")
        plt.ylabel("Distance")

    elif name == "formation_error":  # 编队误差
        plt.figure()
        truncated_data = data_list[:defined_episodes * steps_per_episode]

        # 提取最后一幕的数据
        last_episode_start = (defined_episodes - 1) * steps_per_episode
        last_episode_data = truncated_data[last_episode_start:last_episode_start + steps_per_episode]

        # 提取两个跟随者的误差
        follower1_errors = [d[0] for d in last_episode_data]
        follower2_errors = [d[1] for d in last_episode_data]

        plt.plot(follower1_errors, label='Follower 1')
        plt.plot(follower2_errors, label='Follower 2')
        plt.title("Formation Tracking Error (Last Episode)")
        plt.xlabel("Step in Last Episode")
        plt.ylabel("Error")
        plt.legend()

    elif name == "reward_components":  # 奖励分量绘图
        # 为每个智能体创建图表
        for agent_id, comp_dict in data_list.items():
            plt.figure(figsize=(10, 6))
            # 获取最后一幕的数据
            steps_per_episode = arglist.max_episode_len
            start_idx = (arglist.num_episodes - 1) * steps_per_episode
            end_idx = start_idx + steps_per_episode

            track = comp_dict['track'][start_idx:end_idx]
            collision = comp_dict['collision'][start_idx:end_idx]
            formation = comp_dict['formation'][start_idx:end_idx]

            # 绘制曲线
            steps = range(len(track))
            plt.plot(steps, track, label='track_reward', color='blue', linewidth=2)
            plt.plot(steps, collision, label='collision_reward', color='red', linewidth=2)
            plt.plot(steps, formation, label='formation_reward', color='green', linewidth=2)

            # 设置图表属性
            plt.title(f"{agent_id} - reward_component_last_episode", fontsize=14)
            plt.xlabel("steps", fontsize=12)
            plt.ylabel("reward", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # 保存每个智能体分值奖励图
            save_path = os.path.join(file_dir, f"{name}_{agent_id}.png")
            plt.savefig(save_path)

            # 关闭当前图表，避免内存漏存
            plt.show()
            # plt.close()
    # 图表显示
    if name != "reward_components":
        save_path = file_dir + name + '.png'
        plt.savefig(save_path)
        plt.show()

    # # 图表关闭
    # if name != "reward_components":
    #     plt.close()

def load_data3_plot(agent_name, file_dir, exp_name, show=True):
    # 加载累积奖励和平均奖励数据
    cumulative_file = os.path.join(file_dir, f"{exp_name}_{agent_name}_cumulative_rew.pkl")
    average_file = os.path.join(file_dir, f"{exp_name}_{agent_name}_average_rew.pkl")

    with open(cumulative_file, 'rb') as f:
        cumulative_data = pickle.load(f)
    with open(average_file, 'rb') as f:
        average_data = pickle.load(f)

    # 绘制曲线
    plt.figure()
    plt.plot(cumulative_data, label='Cumulative Reward', color='blue')
    plt.plot(average_data, label='Average Reward (Window=100)', color='orange')
    plt.title(f"{agent_name} - Reward Curves")
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.legend()

    save_path = os.path.join(file_dir, f"{exp_name}_{agent_name}_rewards.png")
    plt.savefig(save_path)
    plt.close()

def save_config_2yaml(arglist, exp_time):
    # 保存配置文件
    save_exp_dir = arglist.save_dir + arglist.exp_name + '/' + exp_time + '/'
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    with open(save_exp_dir + 'config.yaml', 'w') as f:
        for key, value in arglist.__dict__.items():
            f.write(f'{key}: {value}\n')