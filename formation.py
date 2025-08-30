from typing import Any

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # world characteristics
        world.dim_c = 2 # 通道维度为2维
        world.collaborative = False #合作模式，共享奖励
        num_agents = 3 # 智能体数目为3个（1个leader,2个follower）
        world.num_agents = num_agents
        num_landmarks = num_agents - 1 #地标数量
        # adding agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name =  'agent %d' % i
            agent.collide = False    # agent与其他agent不碰撞
            agent.silent = True      # agent之间无法发送通信信号
            agent.size = 0.5        # agent的大小
        # adding landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False #静态障碍物
            landmark.size = 0.2
        # Initial Conditions
        self.reset_world(world)
        world.scenario = self
        return world
    #============================重置环境==============================#
    def reset_world(self, world):             # 在每一个episode之前重置世界内各个实体的属性
        # Landmarks characteristics
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.15, 0.15])        # 颜色
            landmark.state.p_vel = np.zeros(world.dim_p)         # 速度
            if i == 0: #目标地标（第一个地标是目标）
                goal = world. landmarks[i]
                goal.color = np.array([0.15, 0.65, 0.15])
                goal.state.p_pos = [0, 0]   #目标颜色和位置信息
                world.agents[0].goal_a = goal
            else:     # 障碍物地标
                if not hasattr(landmark, 'initialized'):
                    # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                    landmark.state.p_pos = [20,-10] #障碍物位置固定
                    landmark.initialized = True  # 标记已初始化

        # Leader characteristics
        world.agents[0].color = np.array([0.85, 0.35, 0.35])
        world.agents[0].adversary = False
        # Followers characteristics
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
            world.agents[i].adversary = False
        #智能体初始位置设置
        ##############################方案1##########################
        # # 起点随机在目标附近的小范围内
        #     center = np.array([0.8,0.6])
        #     for agent in world.agents:
        #         agent.state.p_pos = center + np.random.uniform(-0.3,0.3,world.dim_p)
        #         agent.state.p_vel = np.zeros(world.dim_p)
        # # Random intial states 随机初始化领航者和跟随者的状态信息
        # # for agent in world.agents:
        # #     agent.state.p_pos = np.random.uniform(0.1, 0.9, world.dim_p)
        # #     agent.state.p_vel = np.zeros(world.dim_p)
        #         agent.state.c = np.zeros(world.dim_c)
        ############################方案2###########################
        # 为每个智能体设置特定的初始位置区域
        centers = [
            np.array([60, 55]),   # 领航者中心点
            np.array([63, 59]),  # 第一个跟随者中心点
            np.array([64, 52])   # 第二个跟随者中心点
        ]
        radius = 0.01  # 初始位置区域的半径

        for i, agent in enumerate(world.agents):
            # 在指定圆内随机生成位置
            angle = np.random.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(np.random.uniform(0, 1))
            x = centers[i][0] + r * np.cos(angle)
            y = centers[i][1] + r * np.sin(angle)
            agent.state.p_pos = np.array([x, y])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
       
    def benchmark_data(self, agent, world):
        # returning data for benchmark purposes
        if agent == world.agents[0]: #如果该智能体是领航者，返回其与目标的距离
            return np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos) #返回领航者与目标的距离
        else: #跟随者返回（与障碍物距离）+（与领航者距离）
            dists = []
            for l in world.landmarks[1:]: #排除目标地标，仅考虑障碍地标
                dists.append(np.linalg.norm(agent.state.p_pos - l.state.p_pos)) # 跟随者与障碍物之间距离
            dists.append(np.linalg.norm(agent.state.p_pos - world.agents[0].state.p_pos)) #跟随者与领航者之间距离
            return tuple(dists)
   #======================================奖励函数设置========================#
    def reward(self, agent, world):
        if agent == world.agents[0]: #领航者奖励计算
            #==========1：向目标点收敛奖励===========#
            reward_track =  self.goal_reached(agent, world)
            #==========2：避障奖励=================#
            # reward_collision =  self.collision(agent, world)
            reward_collision = 0
            #==========3：编队奖励================#
            # reward_formation =  self.formation_maintance(agent, world)
            reward_formation =  0
        else: #跟随者奖励计算
            #==========1：跟踪领航者奖励===========#
            # reward_track =  self.leader_track(agent, world)
            reward_track =  0
            #==========2：避障奖励=================#
            # reward_collision =  self.collision(agent, world)
            reward_collision = 0
            #==========3：编队奖励================#
            reward_formation =  self.formation_maintance(agent, world)

        # 总编队奖励
        reward = (reward_collision + reward_track + reward_formation)
        return float(reward)

    #=================================（读取）获取奖励分量函数==========================#
    def get_reward_components(self, agent, world):
        # 计算碰撞奖励分量
        collision_reward = self.collision(agent, world)

        # 计算跟踪奖励分量
        if agent == world.agents[0]:  # 领航者
            track_reward = self.goal_reached(agent, world)
            formation_reward = 0
        else:  # 跟随者
            # track_reward = self.leader_track(agent, world)
            track_reward = 0.1
            formation_reward = self.formation_maintance(agent, world)

        return {
            'track': float(track_reward),
            'collision': float(collision_reward),
            'formation': float(formation_reward)
        }

    #===============领航者向目标点收敛奖励==========================#
    def goal_reached(self, agent, world):
        delta_l = 5 # 纵轴放大系数
        lambda_l = 1.5 # 横轴缩放系数（值越大，曲线越平缓，变化越不明显；值越小，曲线越陡峭，变化越明显）
        k_goal = 0.6
        goal_dist = np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)

        if goal_dist <= 0.3:
            reward_track = 50
        else:
            # reward_track = delta_l * np.exp(- lambda_l * goal_dist) # 纯指数函数形式一
            # reward_track = - delta_l * np.exp( lambda_l * goal_dist) # 纯指数函数形式二
            # reward_track = self.leader_to_goal_reward(goal_dist) # 分段函数形式
            reward_track = - k_goal * goal_dist # 线性函数形式
            # reward_track = delta_l / (0.5 + goal_dist) # 反比例函数形式
        return float(reward_track)
    #===============跟随者跟踪领航者奖励===========================#
    # def leader_track(self, agent, world):
    #     desired_dist = 0.4
    #     delta_f = 4
    #     lambda_f = 1.4
    #     leader_dist = np.linalg.norm(agent.state.p_pos - world.agents[0].state.p_pos)
    #     track_error = abs(leader_dist - desired_dist)
    #
    #     reward_track = - delta_f * np.exp(lambda_f * track_error)    # 纯指数函数
    #     # reward_track = - lambda_f * (leader_dist - desired_dist) ** 2   # 二次函数
    #
    #     return float(reward_track)
    #============================编队奖励函数================================#
    def formation_maintance(self, agent, world):
        #####=================1.直接距离加角度==============================#####
        # # 1：距离约束
        # desired_dist = 5
        # leader_dist = np.linalg.norm(agent.state.p_pos - world.agents[0].state.p_pos)
        # dist_error =  abs(leader_dist - desired_dist)
        # #
        # # # 2：角度约束
        # if agent == world.agents[1]:
        #     desired_angle = 1 * np.pi / 5
        # elif agent == world.agents[2]:
        #     desired_angle =  - 1 * np.pi / 5
        #
        # x_d = world.agents[0].state.p_pos[0] + desired_dist * np.cos(desired_angle)
        # y_d = world.agents[0].state.p_pos[1] + desired_dist * np.sin(desired_angle)
        #
        # error_formation = np.sqrt(np.square(x_d - agent.state.p_pos[0]) + np.square(y_d - agent.state.p_pos[1]))
        #
        # reward_formation = - (abs(x_d - agent.state.p_pos[0]) + abs(y_d - agent.state.p_pos[1]))
        #---------------使用速度矢量计算角度------------------#
        # leader_vel = world.agents[0].state.p_vel
        # leader_vel_norm = leader_vel / (np.linalg.norm(leader_vel) + 1e-5)
        #
        # relative_pos = agent.state.p_pos - world.agents[0].state.p_pos
        # relative_pos_norm = relative_pos / (np.linalg.norm(relative_pos) + 1e-5)
        #
        # # 计算角度误差（使用点积）
        # cos_angle = np.dot(leader_vel_norm, relative_pos_norm)
        # angle_error = np.arccos(np.clip(cos_angle, -1, 1))  # 限制在[-1,1]范围内
        #
        # angle_deviation = abs(angle_error - desired_angle)

        # # 组合奖励
        # dist_reward = 2 * np.exp(-1 * dist_error)
        # angle_reward = 2 * np.exp(-1 * angle_deviation)

        # reward_formation = dist_reward + angle_reward
        #===============使用实际位置计算角度===================#
        # delta_y = agent.state.p_pos[1] - world.agents[0].state.p_pos[1]
        # delta_x = agent.state.p_pos[0] - world.agents[0].state.p_pos[0]
        #
        # angle = np.arctan2(delta_y, delta_x)
        #
        #      angle_error =  abs(desired_angle - angle)
        # if agent == world.agents[2]:
        #     angle_error = abs(- desired_angle - angle)
        #
        # reward_formation = - dist_error - angle_error
        #####=====================2.只考虑距离===========================#####
        # # 3：编队保持奖励
        # ############# 方案1 ########################
        #     # leader_pos = world.agents[0].state.p_pos
        #     # current_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - leader_pos)))
        #     # desired_dist = 0.3
        #     # formation_error = np.abs(current_dist - desired_dist)
        #     # reward -= 0.4 * formation_error
        ############### 方案2 ######################
        # # 期望编队边长
        # h = 5
        # # 编队奖励权重
        # # kf1 = kf2 = kf3 = 2.0
        # # 位置向量
        # H1 = world.agents[0].state.p_pos   # 领航者
        # H2 = world.agents[1].state.p_pos   # 跟随者1
        # H3 = world.agents[2].state.p_pos   # 跟随者2
        # # 实际距离
        # d12 = np.linalg.norm(H1 - H2)
        # d13 = np.linalg.norm(H1 - H3)
        # d23 = np.linalg.norm(H2 - H3)
        # # 允许的小误差
        # tol = 0.65
        #
        # # 计算四舍五入到小数点后两位的误差
        # err12 = round(abs(d12 - h), 2)
        # err13 = round(abs(d13 - h), 2)
        # # err23 = round(abs(d23 - h), 2)
        #
        # if agent == world.agents[0]:
        #     # reward_formation = - 5 * (np.exp(err12) + np.exp(err13))
        #     if err12 <= tol and err13 <tol:
        #         reward_formation = 20
        #     else:
        #         reward_formation = -20
        # elif agent == world.agents[1]:
        #     # reward_formation = - 5 * np.exp(err12)
        #     if err12 <= tol:
        #         reward_formation = 20
        #     else:
        #         reward_formation = -20
        # else:
        #     # reward_formation = - 5 * np.exp(err13)
        #     if err13 <= tol:
        #         reward_formation = 20
        #     else:
        #         reward_formation = -20

        # # 精确编队位置和方向约束
        # if err12 <= tol  and H1[0] < H2[0] and H1[1] < H2[1]:
        # # if H1[0] < H2[0] and H1[1] < H2[1]:
        #     # r_f1 = -2.5 * np.exp(abs(tol - err12))
        #     r_f1 = 10
        # else:
        #     r_f1 = -10
        #
        # if err13 <= tol and H1[0] < H3[0] and H1[1] > H3[1]:
        # # if H1[0] < H3[0] and H1[1] > H3[1]:
        #     # r_f2 = -2.5 * np.exp(abs(tol - err13))
        #     r_f2 = 10
        # else:
        #     r_f2 = -10
        #
        # if err23 <= tol and H2[0] < H3[0] and H2[1] > H3[1]:
        # # if H2[0] < H3[0] and H2[1] > H3[1]:
        #     # r_f2 = -2.5 * np.exp(abs(tol - err23))
        #     r_f2 = 10
        # else:
        #     r_f3 = -10
        #
        # reward_formation = (r_f1 + r_f2 + r_f3)
        ######==================3.拆分成坐标距离分量===========================##########
        # 位置向量
        H1 = world.agents[0].state.p_pos  # 领航者
        H2 = world.agents[1].state.p_pos  # 跟随者1
        H3 = world.agents[2].state.p_pos  # 跟随者2

        if agent == world.agents[1]:
            F1_lx = H2[0] - H1[0]
            F1_ly = H2[1] - H1[1]  # 跟随者F1实际坐标分量距离
            F1_lxd = 3
            F1_lyd = 4
            # if abs(F1_lx - F1_lxd) <= tol and abs(F1_ly - F1_lyd) <= tol:
            #     reward_formation = 20
            # else:
            #     reward_formation = -20
            e_x = F1_lx - F1_lxd
            e_y = F1_ly - F1_lyd
            # reward_formation = - 0.008 * (e_x ** 2 + e_y ** 2) #误差二次项
            reward_formation = - (abs(e_x) + abs(e_y))  #误差一次项
        if agent == world.agents[2]:
            F2_lx = H3[0] - H1[0]
            F2_ly = H3[1] - H1[1]  # 跟随者F2实际坐标分量距离
            F2_lxd = 4
            F2_lyd = -3
            # if abs(F2_lx - F2_lxd) <= tol and abs(F2_ly - F2_lyd) <= tol:
            #     reward_formation = 20
            # else:
            #     reward_formation = -20
            e_x = F2_lx - F2_lxd
            e_y = F2_ly - F2_lyd
            # reward_formation = - 0.01 * (e_x ** 2 + e_y ** 2) # 误差二次项
            reward_formation = - (abs(e_x) + abs(e_y))  #误差一次项

        # # 使用连续奖励函数，只考虑位置约束
        # # reward_formation = 8 * (np.exp(-5*err12) + np.exp(-5*err13) + np.exp(-5*err23)) # 1：指数形式1
        # # reward_formation = -(err12 ** 2 + err13 ** 2 + err23 ** 2) # 2:二次函数形式
        # reward_formation = - delta_fm * (formation_error) ** 2 # 二次函数形式
        # reward_formation = delta_fm * np.exp(- lambda_fm * formation_error) # 指数函数形式2
        # reward_formation = - (np.exp(err12) + np.exp(err13) + np.exp(err23)) # 指数函数形式3
        return float(reward_formation)

    #===============================避障惩罚设置==========================#
    def collision(self, agent, world):
        col_rew = 0.0
        dist_unsafe = 3 * agent.size  #安全阈值
        r_collision = 0.5  # 撞击惩罚
        # lamda_collision = 0.4
        ################# 1：智能体之间避撞###############
        for ag in world.agents:
            if ag != agent:
                other_agent_dist = np.linalg.norm(ag.state.p_pos - agent.state.p_pos)
                if other_agent_dist < dist_unsafe:
                    col_rew = -r_collision
                # else:
                #     # col_rew += 0.5 * other_agent_dist
                #     col_rew += r_collision
        ################# 2：智能体与障碍物之间避撞###############
        # min_dist = 10 #根据具体环境设置
        for lm in world.landmarks[1:]:  # 排除目标地标
            obs_dist = np.linalg.norm(lm.state.p_pos - agent.state.p_pos)
            if obs_dist < dist_unsafe:
                col_rew -= r_collision
            # else:
            #     # if min_dist > obs_dist:
            #     #     min_dist = obs_dist
            #     #     col_rew += lamda_collision * min_dist
            #     col_rew += r_collision
        return float(col_rew)

    #=============================领航者跟踪目标=========================#
    # def leader_to_goal_reward(self, distance, alpha=1.0, beta=0.5, gamma=1.0, delta=1.5):
    #     """
    #     指数+有界二次项混合奖励函数：
    #     r(d) = - [ alpha * exp(beta*d) + gamma * d^2 / (d^2 + delta^2) ]
    #     参数说明：
    #         distance: 当前距离
    #         alpha: 指数项权重
    #         beta: 远距离梯度陡峭程度
    #         gamma: 二次项贡献强度
    #         delta: 软阈值，控制过渡位置
    #     """
    #     exp_term = np.exp(beta * distance)
    #     quad_term = (distance ** 2) / (distance ** 2 + delta ** 2)
    #     return -(alpha * exp_term + gamma * quad_term)

    #==============================终止运行条件===============#
    def done(self, agent, world):
        if agent == world.agents[0]:  # 领航者到达目标区域
            goal_dist = np.linalg.norm(agent.state.p_pos - world.landmarks[0].state.p_pos)
            return goal_dist <= 0.3
        else: #暂时不考虑跟随者单独停止条件，以跟踪领航者为准
            return False

    #==============================智能体出界===========================#
    # def outside(self, agent, world):
    #     out_rew = 0.0
    #     if np.sum(np.absolute(agent.state.p_pos)) > 4.0: # 智能体出界
    #         out_rew -= 10.0
    #     return float(out_rew)

    def observation(self, agent, world):
        if agent == world.agents[0]: # 领航者观测空间
            # 包含自身位置、速度、目标位置、障碍物位置
            goal_pos = [world.landmarks[0].state.p_pos - agent.state.p_pos] # 目标位置
            obstacle_pos = [lm.state.p_pos - agent.state.p_pos for lm in world.landmarks[1:]] # 障碍物位置
            return np.concatenate([agent.state.p_pos, agent.state.p_vel] + goal_pos + obstacle_pos)
            # return np.concatenate([agent.state.p_pos, agent.state.psi, agent.state.u, agent.state.r] + goal_pos + obstacle_pos) # 8维
        else:# 跟随者观测空间
            obs_pos = [lm.state.p_pos - agent.state.p_pos for lm in world.landmarks[1:]]
            # other_pos = []
            # other_vel = []
            # for other in world.agents[:0]:
            #     if other is agent: continue
            #     other_pos.append(other.state.p_pos - agent.state.p_pos)
            #     other_vel.append(other.state.p_vel)
            leader_pos = world.agents[0].state.p_pos - agent.state.p_pos
            return np.concatenate([agent.state.p_pos, agent.state.p_vel] + leader_pos + obs_pos)
            # return np.concatenate([agent.state.p_pos, agent.state.psi, agent.state.u, agent.state.r] + leader_pos + obs_pos) # 8维
        ############## 1. original 观测空间设置 ######################
        # if not agent.adversary:
        #     return np.concatenate([agent.state.p_pos - world.agents[0].state.p_pos] + landmark_pos + other_pos) # 跟随者
        # else:
        #     return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + landmark_pos)  # 领航者
        ############### 2. now 观测空间设置 ######################
        # if not agent.adversary:
        #     return np.concatenate([agent.state.p_pos] + landmark_pos + other_pos) # 跟随者
        # else:
        #     return np.concatenate([agent.state.p_pos] + landmark_pos) # 领航者
        ############## 3:6.5日观测空间设置#############3
        # return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + landmark_pos + other_pos + other_vel)
