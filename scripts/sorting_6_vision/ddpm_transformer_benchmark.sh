MUJOCO_GL=egl python run_vision.py --config-name=sorting_6_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_transformer_vision_agent \
              agent_name=ddpm_transformer_vision \
              window_size=5 \
              group=sorting_6_ddpm_transformer_seeds \
              agents.model.model.n_timesteps=16
