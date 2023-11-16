python run.py --config-name=stacking_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_agent \
              agent_name=ddpm \
              window_size=1 \
              group=stacking_ddpm_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=60 \
              simulation.n_trajectories_per_context=18 \
              agents.model.model.t_dim=4 \
              agents.model.n_timesteps=4