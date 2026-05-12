import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNDetection(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(12, 128)  # ✅ 6 raw + 6 symmetrical component features
        self.lif1 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(128, 1)
        self.lif2 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(mem2)

        return torch.stack(spk2_rec)