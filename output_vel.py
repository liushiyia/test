import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tensorflow.python.ops.gen_batch_ops import batch

# 1.加载速度数据
with open('/home/zyw/code/Critic_Actor_tf2/results/maddpg/learning_curves/2025-08-14-03-59-56/maddpg_velocities.pkl','rb') as f:
    data = pickle.load(f)
    # TODO FOR DEBUG 打印输出data变量类型
    # print(f"数据类型：, {type(data)}")
    # if isinstance(data, dict):
    #     print(f"字典键: {list(data.keys())}")
    #     print(f"第一个键的数据类型: {type(list(data.keys())[0]) if data else '空'}")
    #     print(f"第一个值的数据类型: {type(list(data.values())[0]) if data else '空'}")
    #print(data)

# 2.提取三个智能体对应速度数据
agents = ['agent_0', 'agent_1', 'agent_2']
agent_data = {agent: data[agent] for agent in agents if agent in data}

# 3. 创建输出目录
output_dir = '/home/zyw/code/Critic_Actor_tf2/results/maddpg/learning_curves/2025-08-14-03-59-56'
os.makedirs(output_dir, exist_ok=True)

episode_step = 100 # 每一幕步数
interval_size = 100 # 绘制间隔幕数
# 4. 速度曲线绘制
for agent_id, episodes in agent_data.items():
    n_episodes = len(episodes)
    vx_all = []
    vy_all = []
    steps = []
    episode_count = []
    num = 0

    # 绘制每一幕速度曲线
    for episode_idx, episode in enumerate(episodes):
        episode_count.append(episode_idx)
        filted_episode = [v for v in episode if v is not None and isinstance(v,(tuple))]
        # 提取速度数据
        start_idx = episode_idx * episode_step
        end_idx = start_idx + episode_step
        episode_slice = filted_episode[start_idx:end_idx]

        vx = [v[0] for v in episode_slice]
        vy = [v[1] for v in episode_slice]

        # 构建连续坐标
        if episode_idx % interval_size == 0:
            start_step = num * episode_step
            end_step = start_step + len(episode_slice)
            steps.extend(range(start_step, end_step))

            # 数据合并
            vx_all.extend(vx)
            vy_all.extend(vy)

            num +=1

    # 绘制速度曲线
    plt.figure()
    plt.plot(steps, vx_all, 'b-', label='Vx', linewidth=1, alpha=0.8)
    plt.plot(steps, vy_all, 'g-', label='Vy', linewidth=1, alpha=0.8)

    # plt.show()

    # 图表保存
    plt.savefig(os.path.join(output_dir, f"{agent_id}.png"))
    # plt.close()
























