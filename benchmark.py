import torch
import gc

def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,} ({total_params * 4 / (1024 * 1024):.2f} MB)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params * 4 / (1024 * 1024):.2f} MB)")
    print(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params * 4 / (1024 * 1024):.2f} MB)")
    
    return total_params, trainable_params, non_trainable_params

def measure_memory_usage(model, sample_input=None, sample_target=None, loss_fn=None):
    """
    Measure memory usage during forward and backward passes.
    
    Args:
        model: PyTorch model
        sample_input: Sample input tensor
        sample_target: Sample target tensor (optional, for backward pass)
        loss_fn: Loss function (optional, for backward pass)
        
    Returns:
        dict: Memory usage statistics
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Memory measurement requires CUDA.")
        return {}
    
    device = next(model.parameters()).device
    if not device.type == 'cuda':
        model = model.cuda()
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    # Baseline memory
    baseline = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Forward pass
    if sample_input is not None:
        with torch.no_grad():
            _ = model(sample_input)
        
        forward_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        forward_used = forward_peak - baseline
        print(f"Forward pass memory: {forward_used:.2f} MB")
        
        # Backward pass (if applicable)
        if sample_target is not None and loss_fn is not None:
            torch.cuda.reset_peak_memory_stats()
            output = model(sample_input)
            loss = loss_fn(output, sample_target)
            loss.backward()
            
            backward_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Backward pass memory: {backward_peak:.2f} MB")
            
            return {
                "forward_mb": forward_used,
                "backward_mb": backward_peak
            }
        
        return {"forward_mb": forward_used}
    
    return {}

# Example usage

if __name__ == "__main__":
    from model import VideoClassify
    import torch.nn as nn
    
    model = VideoClassify(num_classes=10, frames=75)
    
    # Get parameter counts
    count_parameters(model)
    
    # Measure memory usage with sample input
    sample_input = torch.randn(1, 75, 3, 128, 128).cuda()
    sample_target = torch.randint(0, 10, (1,)).cuda()
    loss_fn = nn.CrossEntropyLoss()
    
    measure_memory_usage(model, sample_input, sample_target, loss_fn)
