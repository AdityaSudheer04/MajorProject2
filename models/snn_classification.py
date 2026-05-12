import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNClassification(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.fc1 = nn.Linear(12, 256)  # 6 raw + 6 symmetrical component features
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(alpha=2.0))

        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=0.85, spike_grad=surrogate.atan(alpha=2.0))

        # Removed fc3/lif3 (128→64) — going directly to output
        # Width (256→128) still gives enough representational capacity
        self.fc3 = nn.Linear(128, num_classes)
        self.lif3 = snn.Leaky(beta=0.75, spike_grad=surrogate.atan(alpha=2.0))

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)  # Binary spikes

        # Sum spikes across time → spike count per class (batch, classes)
        spk_out = torch.stack(spk3_rec).sum(dim=0)

        # Final membrane potential at last timestep (batch, classes)
        mem_out = mem3

        return spk_out, mem_out