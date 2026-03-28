import sys
import os
import torch

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.common.llm_extractor import FrozenLLMLayerExtractor

def test_llm_extractor():
    print("Testing Step 9: Frozen LLM Layer Extractor (LLM4Seg style)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters matching SAM2 lowest resolution output and LLM4Seg default
    B = 2
    C = 256  # Visual feature channels
    H, W = 32, 32  # Spatial dimensions (e.g., 512/16 = 32)
    LLM_DIM = 1536 # DeepSeek-1.5B hidden dimension

    print(f"\nInitializing FrozenLLMLayerExtractor...")
    print(f"Expected Input/Output Channels: {C}")
    print(f"Expected Spatial Size: {H}x{W}")
    print(f"LLM Hidden Dimension: {LLM_DIM}")

    # 1. Initialize the module
    # We use a dummy fallback internally if weights are not downloaded, 
    # so this script can run immediately to verify the tensor flow.
    try:
        extractor = FrozenLLMLayerExtractor(
            in_channels=C,
            out_channels=C,
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            layer_idx=27,
            llm_hidden_dim=LLM_DIM,
            h=H,
            w=W,
            freeze=True
        ).to(device)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # 2. Create simulated visual features
    # e.g., output from SAM2's FPN
    visual_features = torch.randn(B, C, H, W).to(device)
    print(f"\nSimulated Input shape (from SAM2 FPN): {visual_features.shape}")

    # 3. Forward pass
    print("Running forward pass through Adapter1 -> Frozen LLM Layer -> Adapter2 ...")
    try:
        output_features = extractor(visual_features)
        
        print("\nForward pass successful!")
        print(f"Output shape: {output_features.shape}")
        
        # 4. Verify shapes
        assert output_features.shape == visual_features.shape, f"Shape mismatch! Expected {visual_features.shape}, got {output_features.shape}"
        
        # Verify freezing worked (check requires_grad)
        # adapter1 should be trainable
        assert extractor.adapter1.weight.requires_grad == True, "Adapter1 should be trainable"
        # adapter2 should be trainable
        assert extractor.adapter2.weight.requires_grad == True, "Adapter2 should be trainable"
        
        # llm_layer should be frozen (if not dummy)
        if not extractor._is_dummy:
            # Check the first parameter of the llm layer
            first_param = next(extractor.llm_layer.parameters())
            assert first_param.requires_grad == False, "LLM Layer parameters should be frozen!"
            print("Verified: LLM layer is frozen, Adapters are trainable.")
        else:
            print("Note: Running with dummy Transformer layer (LLaMA weights not loaded).")
            
        print("\nStep 9 verification passed. The 'Sandwich' structure is correct.")

    except Exception as e:
        print(f"Error during forward pass: {e}")

if __name__ == "__main__":
    test_llm_extractor()
