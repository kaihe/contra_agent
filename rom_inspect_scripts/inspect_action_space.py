import stable_retro as retro
import numpy as np

def main():
    GAME = "ContraForce-Nes-v0"
    print(f"Loading {GAME} with retro.Actions.FILTERED...")
    
    # Matches train.py configuration
    env = retro.make(
        game=GAME, 
        use_restricted_actions=retro.Actions.FILTERED,
        render_mode=None
    )
    
    print("\nAction Space Info:")
    print(f"Type: {env.action_space}")
    
    if hasattr(env.action_space, 'n'):
        print(f"Number of actions: {env.action_space.n}")
        
    print("\nButtons List:")
    if hasattr(env, 'buttons'):
        print(env.buttons)
    else:
        print("env.buttons not found")

    print("\nUnwrapped Buttons:")
    if hasattr(env.unwrapped, 'buttons'):
        print(env.unwrapped.buttons)
    else:
        print("env.unwrapped.buttons not found")

    env.close()

if __name__ == "__main__":
    main()
