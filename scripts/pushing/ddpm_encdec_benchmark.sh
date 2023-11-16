python run.py --config-name=pushing_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ddpm_encdec \
              agent_name=ddpm_encdec \
              window_size=8 \
              group=aligning_ddpm_encdec_seeds \
              simulation.n_cores=30 \
              simulation.n_contexts=30 \
              simulation.n_trajectories_per_context=16 \
              agents.model.n_timesteps=16