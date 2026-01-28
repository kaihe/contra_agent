import stable_retro as retro
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    # Load the game
    GAME = "ContraForce-Nes-v0" 
    STATE = "Level1"

    print(f"Loading {GAME}...")
    try:
        env = retro.make(game=GAME, state=STATE, render_mode=None)
    except Exception as e:
        print(f"Error loading game: {e}")
        return

    print("Environment created.")
    
    # Reset environment
    ret = env.reset()
    if isinstance(ret, tuple):
        obs, _ = ret
    else:
        obs = ret
    
    # Setup Matplotlib viewer
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(obs)
    plt.title("Contra Force - Inspect x_pos")
    
    # Action: Move Right
    # Standard NES: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
    # We want RIGHT (index 7)
    action_size = env.action_space.shape[0]
    action = np.zeros(action_size, dtype=int)
    action[7] = 1 # Set Right button
    
    print("Moving RIGHT and tracking x_pos...")
    print("Step | x_pos | Reward")
    print("-" * 30)

    try:
        for i in range(1000):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            x_pos = info.get('x_pos', 'N/A')
            
            # Print info every 10 steps to reduce clutter
            if i % 10 == 0:
                print(f"{i:4d} | {x_pos} | {reward}")
            
            # Render every frame (or skip frames for speed)
            im.set_data(obs)
            fig.canvas.flush_events()
            # plt.pause(0.001) # Short pause to update window
            
            if done:
                print("Episode finished.")
                ret = env.reset()
                if isinstance(ret, tuple):
                    obs, _ = ret
                else:
                    obs = ret
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        plt.close()
        print("Done.")

if __name__ == "__main__":
    main()
