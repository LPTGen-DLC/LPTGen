from typing import List, Optional

from torch.distributions.constraints import lower_cholesky


class Operator:
    def __init__(self, name: str, parameters: List[str], returns: List[str],
                 signature: str, doc: Optional[str] = None,
                 lower_func: Optional= None,
                 lower_conditions: Optional[List[str]] = None,
                 seed_config_list: Optional[List]=None):
        self.name = name
        self.parameters = parameters
        self.returns = returns
        self.signature = signature
        self.doc = doc
        self.lower_func = lower_func
        self.lower_conditions = lower_conditions
        self.seed_config_list = seed_config_list

    def set_seeds(self, seed_config_list):
        self.seed_config_list = seed_config_list


    def __repr__(self):
        return f"Operator(name={self.name}, params={self.parameters}, returns={self.returns})"

