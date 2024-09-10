import torch

class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(in_features=128, out_features=768),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=768, out_features=768),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=768, out_features=128),
            )
    
        def forward(self, x):
            return self.fc(x)