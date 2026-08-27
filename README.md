# networkai-gym-satellite-power-control
A reinforcement learning project for satellite power control using NetworkAI-Gym and DQN.

# RL4Life final project

## How to run code
1. Create a virtual environment with Anaconda
```
conda create --name <env name> python=3.10
```
2. install dependencies
```
pip install -r requirements.txt
```
3. For training RL agents, run the following:
```
python my_gym.py
```
4. If you want to train agents on different maps, simply modify the loaded `file name` at line 194:
```
... load_cart_file="../environment/pop_data/<file name>")
```
- `user_cart_dict.json`: map data based on real-world distribution
- `user_cart_dict_uniform.json`: population is assumed to distribute across whole map uniformly
- `user_cart_dict_poisson.json`: population is assumed to distribute across whole map following poisson distribution
5. For population generation details, you can check out `pop_data/osmnx_test.py`
6. (optional) to generate population csv file from original pbf file, you need to download this file on this [link](https://drive.google.com/file/d/124muutFBF1CKviRJFF1Ej6CGzLQRw1eL/view?usp=sharing)
and put it into `osm_data` directory
