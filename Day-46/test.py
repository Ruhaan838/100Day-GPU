import torch
import muon_extension as me

G = torch.randn(512, 512).cuda()
X = me.newton_schulz5_cuda(G, 5, 1e-7)
print(X)