import math
import numpy as np

from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class NuplanMiniTrajdataTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(NuplanMiniTrajdataTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/.unified_data_cache"
        self.trajdata_source_train = ["nuplan-mini_train"]
        self.trajdata_source_valid = ["nuplan-mini_val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "nuplan" : "/home/stud/nguyenti/storage/user/datasets/nuplan/",  # Updated to your path
        }

        # for debug
        self.trajdata_rebuild_cache = False

        self.rollout.enabled = True
        self.rollout.save_video = True
        self.rollout.every_n_steps = 10000
        self.rollout.warm_start_n_steps = 0

        # training config - reduced for mini dataset
        self.training.batch_size = 16  # Reduced for mini dataset
        self.training.num_steps = 50000  # Reduced for faster training
        self.training.num_data_workers = 4

        self.save.every_n_steps = 5000
        self.save.best_k = 5

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 8
        self.validation.num_data_workers = 2
        self.validation.every_n_steps = 1000
        self.validation.num_steps_per_epoch = 10

        self.on_ngc = False
        self.logging.terminal_output_to_txt = True
        self.logging.log_tb = True  # Enable tensorboard
        self.logging.log_wandb = False  # Disable wandb for testing
        self.logging.wandb_project_name = "tbsim_nuplan_mini"
        self.logging.log_every_n_steps = 100
        self.logging.flush_every_n_steps = 500


class NuplanMiniTrajdataEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(NuplanMiniTrajdataEnvConfig, self).__init__()

        self.data_generation_params.trajdata_centric = "agent"  # or "scene"
        # which types of agents to include from ['unknown', 'vehicle', 'pedestrian', 'bicycle', 'motorcycle']
        self.data_generation_params.trajdata_only_types = ["vehicle"]
        # which types of agents to predict
        self.data_generation_params.trajdata_predict_types = ["vehicle"]
        # list of scene description filters
        self.data_generation_params.trajdata_scene_desc_contains = None
        # whether or not to include the map in the data
        self.data_generation_params.trajdata_incl_map = True
        # max distance to be considered neighbors
        self.data_generation_params.trajdata_max_agents_distance = 50
        # standardize position and heading for the predicted agent
        self.data_generation_params.trajdata_standardize_data = True

        # NOTE: rasterization info must still be provided even if incl_map=False
        #       since still used for agent states
        # number of semantic layers that will be used (based on which trajdata dataset is being used)
        self.rasterizer.num_sem_layers = 3  # nuPlan typically uses 3 layers
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([0], [1], [2])
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1.0 / 2.0  # 2 px/m
        # where the agent is on the map, (0.0, 0.0) is the center
        self.rasterizer.ego_center = (-0.5, 0.0)

        # max_agent_num (int, optional): The maximum number of agents to include in a batch for scene-centric batching.
        self.data_generation_params.other_agents_num = None

        # max_neighbor_num (int, optional): The maximum number of neighbors to include in a batch for agent-centric batching.
        self.data_generation_params.max_neighbor_num = 10  # Reduced for mini dataset 