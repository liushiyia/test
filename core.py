import numpy as np
import math

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.05
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass, m=pv)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel最大速度和加速度
        self.max_speed = 10
        self.accel = 1
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range速度取值范围
        self.u_range = 10.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# # TODO FOR add USV model start1:
# # 新增USV3DOF_simple类
# class USV3DOF_Simple:
#     """
#     Simplified first-order USV 3DOF dynamics.
#     State: [x, y, psi, u, r].
#     Action: [u_c, r_c] (desired surge speed and yaw rate).
#     Dynamics: du/dt=(u_c-u)/T_u, dr/dt=(r_c-r)/T_r, dpsi/dt=r, dx/dt=ucos(psi), dy/dt=usin(psi).
#     """
#     def init(self, T_u=1.0, T_r=1.0, dt=0.1):
#         self.T_u = T_u  # 时间常数 (纵向速度)
#         self.T_r = T_r  # 时间常数 (偏航率)
#         self.dt = dt  # 仿真步长
#
#     def integrate(self, state, action):
#         """
#         使用欧拉法将状态更新为下一时刻状态
#         state: dict 包含 'x','y','psi','u','r'
#         action: array([u_c, r_c])
#         """
#         # 当前状态
#         u = state['u']
#         r = state['r']
#         psi = state['psi']
#         u_c, r_c = action  # 期望纵速和偏航率
#
#         # 一阶滞后响应更新 (纵向速度和偏航率)
#         du = (u_c - u) / self.T_u
#         dr = (r_c - r) / self.T_r
#         u_new = u + du * self.dt
#         r_new = r + dr * self.dt
#
#         # 更新航向角 (偏航率积分)
#         psi_new = psi + r_new * self.dt
#
#         # 角度归一化，确保 psi 在 [-pi, pi]
#         psi_new = (psi_new + math.pi) % (2 * math.pi) - math.pi
#
#         # 根据当前船速和航向更新位置
#         dx = u_new * math.cos(psi_new) * self.dt
#         dy = u_new * math.sin(psi_new) * self.dt
#         x_new = state['x'] + dx
#         y_new = state['y'] + dy
#
#         # 更新状态字典
#         state['x'] = x_new
#         state['y'] = y_new
#         state['psi'] = psi_new
#         state['u'] = u_new
#         state['r'] = r_new
#         return state

# # TODO FOR add USV model end

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.1  # 从0.25改为0.1，减少速度衰减
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3 # 接触裕度，控制力的平滑过渡范围

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self): #由策略控制的智能体
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self): # 由world控制的智能体
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls 智能体作用力
        p_force = self.apply_action_force(p_force)
        # apply environment forces 环境作用力
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    # 原：质点模型更新智能体状态(将Actor网络输出视为环境力)
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping) # 考虑速度衰减
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt # F=ma
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt #s=s0+vt
    # def integrate_state(self, p_force):
    #     # 质点模型更新智能体状态(将Actor网络输出直接作为速度使用)
    #     for i, entity in enumerate(self.entities):
    #         if not entity.movable: continue
    #         entity.state.p_vel = p_force[i]
    #         if entity.max_speed is not None:
    #             speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #             if speed > entity.max_speed:
    #                 entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
    #                                                               np.square(entity.state.p_vel[1])) * entity.max_speed
    #         entity.state.p_pos += entity.state.p_vel * self.dt
    # # TODO FOR add USV model start2:
    # # now：使用一阶相应模型更新USV状态
    # def integrate_state(self):
    #     """
    #     遍历每个智能体，使用 USV 动力学更新其状态
    #     """
    #     for agent in self.agents:
    #         if not agent.movable:
    #             continue
    #         # 初始化状态字典（假设已在 agent.state 中包含 x,y,psi,u,r）
    #         state = {
    #             'x': agent.state.p_pos[0],
    #             'y': agent.state.p_pos[1],
    #             'psi': agent.state.psi,
    #             'u': agent.state.u,
    #             'r': agent.state.r
    #         }
    #         # 读取动作指令 (u_c, r_c) 存储在 agent.action.u
    #         action = agent.action.u  # 形如 array([u_c, r_c])
    #         # 使用USV模型更新状态
    #         usv = USV3DOF_Simple(T_u = agent.T_u, T_r = agent.T_r, dt = self.dt)
    #         new_state = usv.integrate(state, action)
    #         # 将更新后的状态写回 agent.state
    #         agent.state.p_pos = np.array([new_state['x'], new_state['y']])
    #         agent.state.psi = new_state['psi']
    #         agent.state.u = new_state['u']
    #         agent.state.r = new_state['r']
    #
    # # TODO FOR add USV model end
    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities 用于计算两个实体之间的连续可导的碰撞力
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide): # 实体不会发生碰撞
            return [None, None] # not a collider
        if (entity_a is entity_b): # 同一实体
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos # 1.相对位置向量
        dist = np.sqrt(np.sum(np.square(delta_pos))) # 2.实际距离
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size # 3.最小允许距离
        # 通过softmax penetration平滑函数模拟碰撞力
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k # 4.穿透深度
        force = self.contact_force * delta_pos / dist * penetration # 5.碰撞力矢量
        force_a = +force if entity_a.movable else None # 6.最终作用力
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]