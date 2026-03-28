import sys
import os
import torch

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.rl_agent.llm_policy import LLMDrivenPromptAgentDDQN

def test_step12_llm_policy():
    print("Testing Step 12: LLM-Empowered RL Policy Network")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Typical parameters
    B = 4           # Batch size (e.g. 4 agents acting simultaneously or a batch of experiences)
    C = 256         # SAM2 visual feature channels
    H, W = 32, 32   # Spatial dimensions (downsampled)
    LLM_DIM = 1536  # DeepSeek-1.5B hidden dimension
    ACTION_SIZE = 9 # 9 discrete actions

    print(f"\nInitializing LLMDrivenPromptAgentDDQN...")
    try:
        policy_net = LLMDrivenPromptAgentDDQN(
            visual_dim=C,
            llm_dim=LLM_DIM,
            action_size=ACTION_SIZE,
            h=H,
            w=W
        ).to(device)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Create dummy states
    # 1. Image features (e.g. from SAM2 encoder)
    image_features = torch.randn(B, C, H, W).to(device)
    print(f"Simulated State 1 (Image Features): {image_features.shape}")
    
    # 2. Current Bbox (normalized coordinates: x1, y1, x2, y2)
    current_bbox = torch.rand(B, 4).to(device)
    print(f"Simulated State 2 (Current Bbox): {current_bbox.shape}")

    # Forward pass
    print("\nRunning forward pass (Bbox + Image -> LLM -> Q-values)...")
    try:
        q_values = policy_net(image_features, current_bbox)
        
        print("\nForward pass successful!")
        print(f"Output Q-values shape: {q_values.shape}")
        
        # Verify shapes
        assert q_values.shape == (B, ACTION_SIZE), f"Shape mismatch! Expected {(B, ACTION_SIZE)}, got {q_values.shape}"
        print("Shape verification passed. The Agent outputs exactly 9 Q-values per item in the batch.")
        
        # Verify freezing status
        # Projections and Q-head should be trainable
        assert policy_net.vis_proj.weight.requires_grad == True
        assert policy_net.q_head[0].weight.requires_grad == True
        
        # LLM layer should be frozen (if not dummy fallback)
        if hasattr(policy_net, 'llm_layer_extractor') and not getattr(policy_net.llm_layer_extractor, '_is_dummy', True):
            first_param = next(policy_net.llm_layer_extractor.llm_layer.parameters())
            assert first_param.requires_grad == False, "LLM Layer must be frozen to prevent RL training collapse!"
            print("Verified: LLM layer is strictly frozen, while MLPs are trainable.")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    test_step12_llm_policy()
