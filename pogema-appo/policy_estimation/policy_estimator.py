from pathlib import Path

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.utils.utils import log, AttrDict

import torch

from estimate_policy import EstimatorSettings
from policy_estimation.model import PolicyEstimationModel


class PolicyEstimator:
    def __init__(self, cfg=EstimatorSettings()):
        if not torch.cuda.is_available():
            log.warning('No cuda device is available, so using cpu instead!')
            cfg.device = 'cpu'
        self.cfg = cfg

        self._pe = PolicyEstimationModel()
        self._pe.to(cfg.device)

    def load(self, path):
        path = self.get_checkpoints(path)[-1]
        log.warning(f'Loading Policy Evaluation state from checkpoint {path}')
        checkpoint_dict = torch.load(path, map_location=self.cfg.device)
        self._pe.load_state_dict(checkpoint_dict)

    def predict(self, observations):
        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(observations))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(self.cfg.device).float()
            return self._pe(obs_torch).cpu().numpy()

    def get_checkpoints(self, path):
        checkpoints = Path(path).glob('*.pth')
        return sorted(checkpoints)
