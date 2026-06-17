---
description: Set up the development environment for EconomicGrasp
---

# Environment Setup

Activate virtualenv:

// turbo

```bash
source /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp/py310/bin/activate
```

## Verify

```bash
python -c "import torch; import MinkowskiEngine; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```