import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNClassification(nn.Module):
    def __init__(self, num_classes=6):  # ✅ configurable — dataset has 6 fault combinations
        super().__init__()

        self.fc1 = nn.Linear(12, 256)  # ✅ 6 raw + 6 symmetrical component features
        # Different beta per layer — lets each layer learn different temporal dynamics
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(alpha=2.0))

        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=0.85, spike_grad=surrogate.atan(alpha=2.0))

        self.fc3 = nn.Linear(128, 64)
        self.lif3 = snn.Leaky(beta=0.8, spike_grad=surrogate.atan(alpha=2.0))

        self.fc4 = nn.Linear(64, num_classes)  # ✅ was hardcoded 4, now dynamic
        # Output layer — gentler surrogate for stable classification gradients
        self.lif4 = snn.Leaky(beta=0.75, spike_grad=surrogate.atan(alpha=2.0))

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []   # ✅ Store actual SPIKES (binary) for spike-count loss
        mem4_rec = []   # ✅ Store membrane potentials for membrane loss

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)   # ✅ Binary spikes — NOT mem4
            mem4_rec.append(mem4)   # ✅ Membrane potential at each timestep

        # Sum spikes across time → spike count per class (batch, classes)
        spk_out = torch.stack(spk4_rec).sum(dim=0)

        # Final membrane potential at LAST time step (batch, classes)
        mem_out = mem4

        return spk_out, mem_out
