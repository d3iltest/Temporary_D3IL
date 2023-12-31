python run.py --config-name=aligning_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_transformer_agent \
              agent_name=ddpm_transformer \
              window_size=1 \
              group=aligning_ddpm_transformer_seeds_window_1 \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=8 \
              agents.model.n_timesteps=16