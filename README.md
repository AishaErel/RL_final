## Project Structure

```text
rl_final/
├── scripts/
│   └── rl_final/
│       ├── __init__.py
│       ├── lane_change_env.py
│       ├── neighbors.py
│       ├── communication.py
│       ├── world_model.py
│       ├── planner.py
│       ├── request_policy.py
│       └── baselines.py
│
├── train/
│   ├── test_environment.py
│   ├── train_world_model.py
│   ├── train_request_policy.py
│   └── evaluate.py
│
├── notebooks/
│   └── train_and_test_colab.ipynb
│
├── results/
├── checkpoints/
├── requirements.txt
└── README.md
