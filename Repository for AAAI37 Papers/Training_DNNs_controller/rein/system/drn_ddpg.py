"""
This scripts describes how to inherit from the DrnSystems and initialize DRL agent and trainer to build whole
training/evaluation system.
In this project we use DDPG algorithms
https://arxiv.org/abs/1509.02971
DrnSystem includes:
1. Physics: We describe the system dynamics in DroneGrid
2. Agent: we define the structure and parameters of DNNs.
3. Trainer: In the trainer, we describe sampling strategy/loss/gradients/ calculation etc.
3. Reward function: We calculate the reward function in the interaction loop instead of in Physics.
4. Monitor: We define the evaluation metrics here. And use it for logging training and testing processes.

"""


from rein.agent.ddpg import DDPGAgent, DDPGAgentParams
from rein.trainer.trainer_ddpg import DDPGTrainer, DDPGTrainerParams
from rein.system.drn import DrnSystem, DrnSystemParams
from utils import write_config


class DrnDDPGParams(DrnSystemParams):
    """ Here we define a parameter class.
        We save the parameters and their values in a .json file to make parameters tuning easier.
        When we run training/testing script, we need to load the corresponding .json file.
    """

    def __init__(self):
        super().__init__()
        self.agent_params = DDPGAgentParams()
        self.trainer_params = DDPGTrainerParams()


class DrnDDPG(DrnSystem):
    def __init__(self, params: DrnDDPGParams):
        super().__init__(params)
        self.params = params
        if self.params.agent_params.add_actions_observations:
            self.shape_observations_drone += self.params.agent_params.action_observations_dim
        # Initialize DDPG agent
        self.agent = DDPGAgent(params.agent_params, self.shape_observations_drone, shape_action=4)

        # Initialize DDPG Trainer
        self.trainer = DDPGTrainer(params.trainer_params, self.agent)
        self.agent.initial_model()
        if self.params.stats_params.weights_path is not None:
            self.agent.load_weights(self.params.stats_params.weights_path)

        write_config(params, f"{self.model_stats.log_dir}/config.json")

    def test(self):
        self.evaluation_episode(self.agent)
        # self.agent.actor.save(self.params.stats_params.model_name)

