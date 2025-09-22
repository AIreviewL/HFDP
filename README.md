

# 🚀 Human Feedback Hierarchical Diffusion Policy (HFDP)

**HFDP** is a hierarchical framework for long-horizon bimanual manipulation.  
It addresses the limitations of diffusion policies in long-horizon tasks—such as error accumulation and trajectory instability—by combining:

- **LLM-based task decomposition** for better long-horizon planning,  
- **Human feedback** to dynamically correct diffusion-generated trajectories,  
- **Option-Critic framework** to enable hierarchical reinforcement learning and interpretable option selection,  
- **Learning-based inverse kinematics and impedance control** for stable low-level execution.  

🔗 Project Page: [https://sites.google.com/view/hf-dp](https://sites.google.com/view/hf-dp)

## 🛠️ Installation & Usage

### Environment Requirements

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy scipy transforms3d
pip install pyyaml openai

# RoboTwin original dependencies
# Please refer to the official RoboTwin installation guide
```

## 🔧 Configuration

```yaml
# Model configuration
models:
  multimodal_encoder:
    language_dim: 768      # Dimension of language features
    vision_dim: 512        # Dimension of vision features
    robot_state_dim: 64    # Dimension of robot state features
    tactile_dim: 32        # Dimension of tactile features
    
  ik_impedance:
    joint_dim: 12          # Number of robot joints
    cartesian_dim: 6       # Cartesian space dimensions
    
# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  loss_weights:
    ik_loss: 1.0
    impedance_loss: 0.5
    reward_loss: 0.3
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
    # Reduce batch size or switch to CPU
    config['device'] = 'cpu'
    config['batch_size'] = 1
   ```

2. **LLM API Connection Failure**
   ```python
    # Verify API key configuration
    # Set the correct API key in code_gen/gpt_agent.py
   ```

3. **Model Loading Failure**
   ```python
    # Check model file paths
    # Ensure all dependent modules are properly imported
   ```

### Debug Mode
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logs
config['logging_level'] = 'DEBUG'
```
