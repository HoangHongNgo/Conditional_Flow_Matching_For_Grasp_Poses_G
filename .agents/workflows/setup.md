---
description: Set up the development environment for EconomicGrasp
---

# Environment Setup

Use the project Python binary directly from `py310`. Do not use plain `python`.

## Verify

```bash
cd /media/dsp520/Grasp_2T/EconomicGrasp/EconomicGrasp
./py310/bin/python -c "import torch; import MinkowskiEngine; print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())"
```
