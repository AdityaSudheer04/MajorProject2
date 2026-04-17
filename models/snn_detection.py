import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNDetection(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 256)
        self.lif1 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

        self.fc3 = nn.Linear(128, 64)
        self.lif3 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

        self.fc4 = nn.Linear(64, 1)
        self.lif4 = snn.Leaky(beta=0.8, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []  # ← Change from spk3_rec to spk4_rec

        for t in range(x.size(0)):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc4(spk3)  # ← Add this line
            spk4, mem4 = self.lif4(cur4, mem4)  # ← Add this line
            spk4_rec.append(mem4)  # ← Add this line

        return torch.stack(spk4_rec)  # ← Return spk4_rec