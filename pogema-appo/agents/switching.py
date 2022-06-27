from abc import abstractmethod

import numpy as np
from pydantic import BaseModel, Extra

from agents.epom import EpomConfig, EPOM
from agents.replan import RePlanConfig, RePlan


class SwitcherBaseConfig(BaseModel, extra=Extra.forbid):
    planning: RePlanConfig = RePlanConfig(fix_loops=True, add_none_if_loop=True, no_path_random=False,
                                          use_best_move=False, fix_nones=False)
    learning: EpomConfig = EpomConfig()
    num_process: int = 10
    clear_hidden_after_switch: bool = False


class SwitcherBase:
    def __init__(self, algo_cfg):
        self.cfg = algo_cfg
        self.planning = RePlan(algo_cfg.planning)
        self.learning = EPOM(algo_cfg.learning)

        self.learning_used = 0
        self.planning_used = 0
        self.num_switches = 0
        self._previous_mask = None
        self._previous_hidden = None
        self.hidden_noise = 0.0

    @abstractmethod
    def get_learning_use_mask(self, planning_actions, learning_actions, observations):
        raise NotImplementedError

    def act(self, observations, rewards=None, dones=None, infos=None):
        if infos is None:
            infos = [{'is_active': True} for _ in range(len(observations))]
        planning = self.planning.act(observations, rewards, dones, infos)
        learning = self.learning.act(observations, rewards, dones, infos)
        masks = self.get_learning_use_mask(planning_actions=planning, learning_actions=learning,
                                           observations=observations)

        self.update_usage(masks, infos, dones)
        if self.cfg.clear_hidden_after_switch:
            for agent_idx, mask in enumerate(masks):
                if not mask:
                    self.learning.clear_hidden(agent_idx)

        return [learning[idx] if masks[idx] else planning[idx] for idx, _ in enumerate(masks)]

    def after_step(self, dones):
        self.planning.after_step(dones)
        self.learning.after_step(dones)

    def after_reset(self):
        self.planning.after_reset()
        self.learning.after_reset()
        self._previous_mask = None
        self._previous_hidden = None

    def get_additional_info(self):
        result = {}
        if self.learning_used or self.planning_used:
            result.update(rl_used=self.learning_used / (self.learning_used + self.planning_used))
        result.update(num_switches=self.num_switches)
        result.update(hidden_noise=self.hidden_noise)

        self.learning_used = 0
        self.planning_used = 0
        self.num_switches = 0
        self.hidden_noise = 0.0
        return result

    def update_usage(self, mask, infos, dones):

        for idx, info in enumerate(infos):
            if not info['is_active']:
                continue
            if mask[idx]:
                self.learning_used += 1
            else:
                self.planning_used += 1

        # current_hidden = self.learning.rnn_states.cpu().numpy()
        # avg_noise = 0
        # if self._previous_mask and self._previous_hidden:
        #     for agent_idx, (current_mask, previous_mask) in enumerate(zip(mask, self._previous_mask)):
        #         if current_mask != previous_mask:
        #             self.num_switches += 1
                # if self._previous_hidden[agent_idx] is not None:
                #     if current_mask != previous_mask and current_mask:
                #         diff = current_hidden[agent_idx] - self._previous_hidden[agent_idx]
                #         avg_noise += np.abs(diff).mean()
                #         self._previous_hidden[agent_idx] = None
        # self.hidden_noise += avg_noise / len(mask)
        # self._previous_mask = mask
        # if self._previous_hidden is None:
        #     self._previous_hidden = [None for _ in range(len(infos))]

        # for agent_idx, m in enumerate(mask):
        #     if m:
        #         self._previous_hidden[agent_idx] = current_hidden[agent_idx]


