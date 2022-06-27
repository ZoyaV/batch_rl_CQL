from pogema import GridConfig
from heapq import heappop, heappush
INF = int(1e+7)


class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0, safe_interval: (int, int) = (0, INF)):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h
        self.safe_interval = safe_interval
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))


class PrioritizedBase:
    def __init__(self):
        gc: GridConfig = GridConfig()
        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}
        self.trajectories = None
        self.step = 0
        self.priorities = list()
        self.replan = True
        self.replans = 0
        self.states = None

    def update_states(self, trajectory):
        for t in range(len(trajectory)):
            n, m = trajectory[t]
            for k in range(len(self.states[n][m])):
                if self.states[n][m][k][0] <= t <= self.states[n][m][k][1]:
                    if self.states[n][m][k][0] == t:
                        self.states[n][m][k][0] = self.states[n][m][k][0] + 1
                    elif self.states[n][m][k][1] == t:
                        self.states[n][m][k][1] = self.states[n][m][k][1] - 1
                    else:
                        if t + 1 <= self.states[n][m][k][1]:
                            new_interval = [t + 1, self.states[n][m][k][1]]
                            self.states[n][m].insert(k + 1, new_interval)
                        self.states[n][m][k][1] = t - 1

    def act(self, obs, skip_agents=None):
        num_agents = len(obs)
        action = list()
        if len(self.priorities) == 0:
            distances = {}
            for i in range(len(obs)):
                distances[i] = abs(obs[i]['global_xy'][0] - obs[i]['global_target_xy'][0]) + abs(obs[i]['global_xy'][1] - obs[i]['global_target_xy'][1])
            self.priorities = [k for k, v in sorted(distances.items(), key=lambda item: item[1])]
        if self.replan:
            self.states = [[[[0, INF]] for j in range(len(obs[0]['global_obstacles'][0]))] for i in range(len(obs[0]['global_obstacles']))]
            for i in range(num_agents):
                if obs[i]['global_xy'] == obs[i]['global_target_xy']:
                    continue
                n, m = obs[i]['global_xy']
                self.states[n][m] = [[1, INF]]
            self.replan = False
            self.step = 0
            self.trajectories = [[obs[agent_id]['global_xy']] if obs[agent_id]['xy'] != obs[agent_id]['global_target_xy'] else (-1, -1) for agent_id in range(num_agents)]
            for i in range(num_agents):
                agent_id = self.priorities[i]
                if obs[agent_id]['global_xy'] == obs[agent_id]['global_target_xy']:
                    continue
                self.trajectories[agent_id] = self.find_trajectory(obs[agent_id]['global_obstacles'], obs[agent_id]['global_xy'], obs[agent_id]['global_target_xy'])
                if self.trajectories[agent_id] is None:
                    self.replan = True
                    self.priorities.insert(0, self.priorities.pop(self.priorities.index(agent_id)))
                    self.replans += 1
                    if self.replans < 100:
                        return self.act(obs, skip_agents)
                    else:
                        self.trajectories[agent_id] = [obs[agent_id]['global_xy']]
                self.update_states(self.trajectories[agent_id])

        self.replans = 0
        for i in range(num_agents):
            if skip_agents and skip_agents[i]:
                action.append(None)
                continue
            if obs[i]['global_xy'] == obs[i]['global_target_xy']:
                action.append(None)
                continue
            if self.trajectories[i][0] != self.trajectories[i][-1] and len(self.trajectories[i]) > self.step+1:
                a = (self.trajectories[i][self.step + 1][0] - self.trajectories[i][self.step][0],
                     self.trajectories[i][self.step + 1][1] - self.trajectories[i][self.step][1])
                action.append(self.actions[a])
            else:
                action.append(0)
        self.step += 1
        return action

    def has_collision(self, s):
        for d in self.trajectories:
            poses = [d[s.g+i] for i in range(-1, min(2, len(d) - s.g))]
            if (s.i, s.j) in poses:
                return True
        return False

    def find_trajectory(self, static_obs, start, goal):
        OPEN = list()
        heappush(OPEN, Node(start, 0, abs(goal[0] - start[0]) + abs(goal[1] - start[1]), self.states[start[0]][start[1]][0]))
        CLOSED = set()
        actions = GridConfig().MOVES[1:]
        u = Node()
        while len(OPEN) > 0 and (u.i, u.j) != goal:
            u = heappop(OPEN)
            if (u.i, u.j, u.safe_interval[0]) in CLOSED:
                continue
            CLOSED.add((u.i, u.j, u.safe_interval[0]))
            neighbors = []
            for a in actions:
                n = (u.i + a[0], u.j + a[1])
                if n[0] < 0 or n[0] >= len(static_obs) or n[1] < 0 or n[1] >= len(static_obs):
                    continue
                if static_obs[n[0]][n[1]] == 0:
                    neighbors.append(n)
            for n in neighbors:
                for safe_interval in self.states[n[0]][n[1]]:
                    if max(u.g, safe_interval[0]) + 1 < safe_interval[1] and u.safe_interval[1] > safe_interval[0]:
                        s = Node(n, max(u.g, safe_interval[0]) + 1, abs(goal[0] - n[0]) + abs(goal[1] - n[1]), safe_interval)
                        if (s.i, s.j, s.safe_interval[0]) in CLOSED:
                            continue
                        s.parent = u
                        heappush(OPEN, s)
        trajectory = None
        if (u.i, u.j) == goal:
            trajectory = list()
            while u.g != 0:
                trajectory.append((u.i, u.j))
                parent = u.parent
                while parent.g + 1 != u.g:
                    trajectory.append((parent.i, parent.j))
                    u.g -= 1
                u = parent
            trajectory.append(start)
            trajectory.reverse()
        return trajectory

