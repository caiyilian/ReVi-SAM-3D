import sys
import os
import torch

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.rl_agent.dqn_agent import DQNAgent

def test_step13_dqn_agent():
    print("Testing Step 13: DQN Agent Skeleton and Replay Buffer")
    
    # 1. Initialize Agent
    agent = DQNAgent(visual_dim=256, llm_dim=1536, action_size=9, buffer_capacity=1000)
    print("\nDQNAgent initialized successfully with LLM Policy Networks!")
    
    # Check freezing
    trainable_count = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in agent.policy_net.parameters() if not p.requires_grad)
    print(f"Trainable parameters (MLPs): {trainable_count:,}")
    print(f"Frozen parameters (LLM Layer): {frozen_count:,}")
    assert frozen_count > trainable_count, "The LLM layer should dominate the parameter count and be frozen!"

    # 2. Simulate Environment Interaction
    print("\nSimulating Agent-Environment Interaction and filling Replay Buffer...")
    
    C, H, W = 256, 32, 32
    # Create fake states
    for i in range(40): # Generate 40 fake experiences
        state_img = torch.randn(C, H, W)
        state_bbox = torch.rand(4)
        
        # Select action
        action = agent.select_action(state_img, state_bbox, is_training=True)
        
        # Simulate environment returning next state and reward
        next_state_img = torch.randn(C, H, W)
        next_state_bbox = torch.rand(4)
        reward = float(torch.randn(1).item())
        done = random.choice([True, False])
        
        # Push to buffer
        agent.replay_buffer.push(state_img, state_bbox, action, reward, next_state_img, next_state_bbox, done)
        
    print(f"Replay Buffer current size: {len(agent.replay_buffer)}")
    assert len(agent.replay_buffer) == 40

    # 3. Simulate Learning Step
    print("\nRunning a DQN Learning Step (Batch Size = 16)...")
    try:
        loss = agent.learn(batch_size=16)
        print(f"Learning Step Successful! Loss: {loss:.4f}")
        print(f"Epsilon decayed to: {agent.epsilon:.4f}")
        print("\nStep 13 verification passed. The Agent is ready to be plugged into the Closed-Loop Environment!")
    except Exception as e:
        print(f"Error during learning step: {e}")

if __name__ == "__main__":
    import random
    test_step13_dqn_agent()
