MUJOCO_GL=egl python run_vision.py --config-name=stacking_vision_config \
              --multirun seed=0,1,2,3,4,5 \
              agents=gpt_vision_agent \
              agent_name=gpt_bc \
              window_size=5 \
              group=stacking_4_gpt_bc_seeds