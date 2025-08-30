import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import numpy as np
from base.args_config import parse_args_maddpg

def plot_traj(data_dir, cur_dir, episode_index, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    traj_file = base_path / f"maddpg_trajectories.pkl"

    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)

    episode_data = traj_data[episode_index - 1]
    plt.figure()

    # 提取当前幕的数据
    episode_num = episode_data['episode']
    trajectories = episode_data['agent_trajectories']
    landmarks = episode_data['landmarks']

    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 不同智能体的颜色
    # TODO FOR DEBUG mark special steps start1:
    mark_steps = [20, 40, 60, 80]  # 需要标记的步数
    markers = ['^', '>', '<', 'v']  # 不同步数对应的三角形方向
    # 存储每个步数的所有智能体位置，用于绘制连接线
    step_positions = {step: [] for step in mark_steps}
    # TODO FOR DEBUG mark special steps end
    # 处理字典类型的轨迹数据
    for i, (agent_id, traj) in enumerate(trajectories.items()):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i % len(colors)],
                 alpha=0.6, label=f'Agent {i} Path')
        plt.scatter(traj[0, 0], traj[0, 1], s=100,
                    marker='o', color=colors[i % len(colors)])
        plt.scatter(traj[-1, 0], traj[-1, 1], s=100,
                    marker='X', color=colors[i % len(colors)])
        # TODO FOR DEBUG mark special steps start2:
        # 在指定步数添加三角形标记
        for step, marker in zip(mark_steps, markers):
            if step < len(traj):  # 确保步数在轨迹范围内
                plt.scatter(traj[step, 0], traj[step, 1], s=80,
                            marker=marker, edgecolor='k',
                            facecolor=colors[i % len(colors)],
                            zorder=5 ) # 确保标记在顶层
                # 存储位置用于连接线
                step_positions[step].append((traj[step, 0], traj[step, 1]))
        # 绘制每个步数的智能体位置连接线
        for step, positions in step_positions.items():
            if len(positions) == len(trajectories):  # 确保所有智能体都有该步数的位置
                # 将位置按顺序连接
                x = [p[0] for p in positions]
                y = [p[1] for p in positions]
                # 添加第一个点使连接线闭合
                x.append(positions[0][0])
                y.append(positions[0][1])
                plt.plot(x, y, 'k--', linewidth=1.5, alpha=0.7)
        # TODO FOR DEBUG mark special steps end
    goal_pos = landmarks[0]
    for i, landmark in enumerate(landmarks[1:]):
        plt.scatter(landmark[0], landmark[1], s=200,
                    marker='X', color='k', label=f'Landmark {i}' if i == 0 else "")
    plt.scatter(goal_pos[0], goal_pos[1], color='red', marker='*', s=150, label='Goal')

    plt.title(f'Agent Trajectories and Landmarks (Episode {episode_num})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()

    save_path = base_path / f"trajectory.png"
    plt.savefig(save_path)
################################################################################################
def plot_vel(data_dir, cur_dir, episode_index, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    velocity_file = base_path / f"maddpg_velocity.pkl"

    with open(velocity_file, 'rb') as f:
        velocity_data = pickle.load(f)

    # 找到指定幕的数据
    episode_data = None
    for data in velocity_data:
        if data['episode'] == episode_index:
            episode_data = data
            break

    if episode_data is None:
        print(f"Episode {episode_index} not found in velocity data")
        return

    # 创建包含三个子图的图表
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    velocity_components = ['Vx', 'Vy']
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # 不同智能体的颜色

    # 提取当前幕的数据
    agent_velocity = episode_data['agent_vel']

    # 绘制每个速度分量
    for comp_idx, comp_name in enumerate(velocity_components):
        ax = axes[comp_idx]

        # 绘制每个智能体的当前速度分量
        for i, (agent_id, velocity_values) in enumerate(agent_velocity.items()):
            comp_values = [v[comp_idx] for v in velocity_values]
            steps = range(len(comp_values))
            ax.plot(steps, comp_values, color=colors[i % len(colors)],
                    label=f'Agent {i} {comp_name}', linewidth=1)

        ax.set_title(f'Velocity Component {comp_name} - Episode {episode_index}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{comp_name} Value')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    save_path = base_path / f"velocity_episode_{episode_index}.png"
    plt.savefig(save_path)
    plt.show()
################################################################################################
def plot_reward_components(arglist,data_dir, cur_dir, episode_index, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    rew_components_file = base_path / f"maddpg_reward_components.pkl"

    with open(rew_components_file, 'rb') as f:
        rew_component_data = pickle.load(f)

    plt.figure()

    for agent_id, comp_dict in rew_component_data.items():
        plt.figure(figsize=(10, 6))
        # 获取指定幕episode_index的数据
        steps_per_episode = arglist.max_episode_len
        start_idx = (episode_index - 1) * steps_per_episode
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
        save_path = base_path / f"{agent_id}.png"
        plt.savefig(save_path)
############################################################################################################
def plot_formation_errors(arglist,data_dir, cur_dir, episode_index, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    track_error_file = base_path / f"maddpg_tracking_errors.pkl"

    with open(track_error_file, 'rb') as f:
        track_error_data = pickle.load(f)

    plt.figure()
    truncated_data = track_error_data[:arglist.num_episodes * arglist.max_episode_len]

    # 提取指定幕episode_index的数据
    last_episode_start = (episode_index - 1) * arglist.max_episode_len
    last_episode_data = truncated_data[last_episode_start:last_episode_start + arglist.max_episode_len]

    # 提取两个跟随者的误差
    follower1_errors = [d[0] for d in last_episode_data]
    follower2_errors = [d[1] for d in last_episode_data]

    plt.plot(follower1_errors, label='Follower 1')
    plt.plot(follower2_errors, label='Follower 2')
    plt.title("Formation Tracking Error (Last Episode)")
    plt.xlabel("Step in Last Episode")
    plt.ylabel("Error")
    plt.legend()

    save_path = base_path / f"formation_error.png"
    plt.savefig(save_path)
#############################################################################################################
def plot_obs_dists(arglist,data_dir, cur_dir, episode_index, show=True):
    # 构建文件路径
    base_path = Path(data_dir) / cur_dir
    obs_dist_file = base_path / f"maddpg_obstacle_dists.pkl"

    with open(obs_dist_file, 'rb') as f:
        obs_dist_data = pickle.load(f)

    plt.figure()
    truncated_data = obs_dist_data[:arglist.num_episodes * arglist.max_episode_len]

    # 提取指定幕episode_index的数据
    last_episode_start = (episode_index - 1) * arglist.max_episode_len
    last_episode_data = truncated_data[last_episode_start:last_episode_start + arglist.max_episode_len]

    plt.plot(last_episode_data)
    plt.title("Minimum Distance to Obstacles (Last Episode)")
    plt.xlabel("Step in Last Episode")
    plt.ylabel("Distance")
    save_path = base_path / f"obs_dist.png"
    plt.savefig(save_path)

if __name__ == '__main__':
    arglist = parse_args_maddpg()
    parser = argparse.ArgumentParser(description='Plot trajectories')
    parser.add_argument("--data_dir", type=str, default= "/home/zyw/code/Critic_Actor_tf2/results/maddpg/learning_curves", help='Directory containing data')
    parser.add_argument("--cur_dir", type=str, default="2025-08-29-19-47-19")
    parser.add_argument("--episode_index", type=int, default="500")
    parser.add_argument("--show", action='store_true', default="True")

    args = parser.parse_args()
    plot_traj(args.data_dir, args.cur_dir, args.episode_index, args.show) # 绘制任意幕智能体运动轨迹
    plot_vel(args.data_dir, args.cur_dir, args.episode_index, args.show)  # 新增：绘制任意幕速度分量
    plot_reward_components(arglist, args.data_dir, args.cur_dir, args.episode_index, args.show)  # 绘制任意幕智能体奖励分量
    plot_formation_errors(arglist, args.data_dir, args.cur_dir, args.episode_index, args.show)  # 绘制任意幕编队误差曲线
    plot_obs_dists(arglist, args.data_dir, args.cur_dir, args.episode_index, args.show)  # 绘制任意幕与障碍物距离变化曲线