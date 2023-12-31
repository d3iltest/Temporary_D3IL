python run.py --config-name=pushing_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=ibc_agent \
              agent_name=ibc \
              window_size=1 \
              group=aligning_ibc_seeds \
              simulation.n_cores=5 \
              simulation.n_contexts=30 \
              simulation.n_trajectories_per_context=16 \
              agents.sampler.sampler_stepsize_init=0.0493