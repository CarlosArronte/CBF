import gym
import yaml
import numpy as np
import torch
import time

from f110_gym.envs.base_classes import Integrator

#CBF
from NN_CBF.mlp2no import MLP2NO
from NN_CBF.cbf_headway_beams import CBF

# Controller
from bc_inference_controller import BCInferenceController


# ============================================================
# CONFIG
# ============================================================
MAP_NAME = "maps/Monza_map"              # o el que uses
MAP_PATH = "maps/Monza_map.yaml"    # path real a tu mapa
MODEL_PATH = "bc_policy.pt"
SCALER_PATH = "bc_scaler.pkl"

SCALER_MEAN_PATH = "bc_scaler_mean.npy"
SCALER_STD_PATH = "bc_scaler_std.npy"

WHEELBASE = 0.33
DT = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RENDER = True
MAX_STEPS = 100_000_000


# ============================================================
# ENV CREATION
# ============================================================
def make_env():
    with open(MAP_PATH, "r") as f:
        map_cfg = yaml.safe_load(f)

    env = gym.make(
        "f110_gym:f110-v0",
        map=MAP_NAME,
        map_ext=".png",
        num_agents=1,
        timestep=DT,
        integrator=Integrator.RK4,
    )

    obs, _, _, _ = env.reset(np.array([[map_cfg["start_x"],
                                        map_cfg["start_y"],
                                        map_cfg["start_theta"]]]))

    return env, obs


# ============================================================
# MAIN
# ============================================================
def main():
    env, obs = make_env()
    filter = MLP2NO()
    barrier = CBF()

    controller = BCInferenceController(
        model_path=MODEL_PATH,
        scaler_mean_path=SCALER_MEAN_PATH,
        scaler_std_path=SCALER_STD_PATH,
        wheelbase=WHEELBASE,
        dt=DT,
        device=DEVICE,
    )

    step = 0
    t0 = time.perf_counter()
    
    last_acc = 0.0
   
    while step < MAX_STEPS:
        #MLP controller
        action = controller.act(obs)      

        current_speed = action[0][1]
        
        dt = DT

        v_meas = float(obs["linear_vels_x"][0])
        raw_acc = np.clip((current_speed - v_meas) / dt, -6.0, 6.0)
        
        tau = 0.01
        alpha = tau / (tau + dt)
        filtered_acc = alpha * last_acc + (1 - alpha) * raw_acc      
        
        # Apply the barrier/filter
        mlp_params = {
            "odometry_speed": v_meas,
            "acceleration": float(filtered_acc),
            "steering": float(action[0][0]),
            "ranges": np.array(obs["scans"][0])
        }
       
        #filter_params = filter.filter_mlp(mlp_params)

        cbf_action = barrier.control(
            mlp_params["ranges"],
            mlp_params["steering"],
            mlp_params["acceleration"],
            mlp_params["odometry_speed"],
        )

        acc_cmd = cbf_action["acceleration"]
        cbf_speed = v_meas + acc_cmd * DT
        cbf_steer = float(cbf_action["steer"])
        # f110_gym expects action with shape (num_agents, 2)
        # Explicit key access avoids relying on dict insertion order.
        env_action = np.array(        
            [[cbf_steer, cbf_speed]],
            dtype=np.float32,
        )
        print("\nDEBUG")
        print(f"Nominal Controller=> Speed:{action[0][1]}, Steer:{action[0][0]}")
        print(f"CBF=> Speed:{cbf_speed}, Steer:{cbf_steer}")
        print(f"MEAS=> Speed:{v_meas}, Acc_cmd:{acc_cmd}")
        
        print(f"-45:{np.array(obs['scans'][0][0])}")
        print(f"0:{np.array(obs['scans'][0][180])}")
        print(f"45:{np.array(obs['scans'][0][180*2])}")
        print(f"90:{np.array(obs['scans'][0][180*3])}")
        print(f"135:{np.array(obs['scans'][0][180*4])}")
        print(f"180:{np.array(obs['scans'][0][180*5])}")
        print(f"225:{np.array(obs['scans'][0][180*6-1])}")
        
        print("END_DEBUG\n")
        obs, reward, done, info = env.step(env_action)

        if RENDER:
            env.render(mode='human')

        if done:
            print(f"[INFO] Episode finished at step {step}")
            break

        step += 1        
        
        last_acc = acc_cmd

    elapsed = time.time() - t0
    print(f"[DONE] {step} steps in {elapsed:.2f}s ({step/elapsed:.1f} Hz)")

    env.close()


# ============================================================
if __name__ == "__main__":
    main()




#BDCBAACBB
