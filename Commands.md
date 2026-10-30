python -m pip install -r requirements-bridge.txt


# Open the simulator
python src/sim_bridge.py 

# Open an automatic open/close motion of the URDF gripper
python tools/preview_parallel_gripper.py

# Create 100 randomized tasks using seed 42:
python src/create_task_suite.py --episodes 100 --seed 42 --output dataset_tasks.json


# record the task suite from another terminal
python src/record_vla_dataset.py --tasks dataset_tasks.json --output vla_dataset_raw --duration 20 --video-fps 10

# Creating 100 randomized tasks makes a JSON file containing 100 pick-and-place episodes. Each task specifies an instruction, which object to move, and starting positions for the movable objects. The generator varies the placements while keeping them within its constraints, so the episodes cover different arrangements.

The seed is the number used to initialize the random generator. Reusing the same seed with the same code produces the same task suite, which makes runs reproducible. Change the seed to generate a different suite.

Recording a task suite means running through the tasks in that JSON file and saving the observations and actions for each episode. With the manual recorder, the bridge resets the scene for each task and records the camera videos, joint state, commanded targets, timestamps, and success label.
