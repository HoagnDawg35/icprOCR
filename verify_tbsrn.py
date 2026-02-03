import torch
from src.models import ResTranOCR
from configs import Config

def test_model():
    print("Testing ResTranOCR with TBSRN...")
    config = Config()
    config.num_classes = 37
    config.use_tbsrn = True
    
    model = ResTranOCR(
        num_classes=config.num_classes,
        num_frames=5,
        use_tbsrn=True,
        sr_config=config.sr_config
    ).cuda()
    
    # Dummy input [B, F, C, H, W]
    x = torch.randn(2, 5, 3, 32, 128).cuda()
    
    print("Running forward pass...")
    logits, sr_out = model(x, return_sr=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"SR Out shape: {sr_out.shape}")
    
    # Updated expectations for lighter model: 32 channels in sr_config
    assert sr_out.shape == (2*5, 3, 64, 256), f"Expected SR shape (10, 3, 64, 256), got {sr_out.shape}"
    print("✅ Model Forward Pass Successful!")

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
