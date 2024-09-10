import random
import torch

def get_rectangle():
    width = random.random()
    height = random.random()

    fat = int(width >= height)

    return width, height, fat

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 500
    
    def __getitem__(self, i):
        width, height, fat = get_rectangle()

        x = torch.FloatTensor([width, height])
        y = fat

        return x,y

dataset = Dataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


def train():
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fun = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(100):

        for i, (x, y) in enumerate(loader):

            out = model(x)

            loss = loss_fun(out, y)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        if(epoch % 20 == 0):
            acc = (out.argmax(dim=1) == y).sum().item()
            print(f"epoch: {epoch}, loss: {loss.item()}, acc {acc}")
    
    torch.save(model, 'lansinuote/model/ractangle.model')

@torch.no_grad()
def test():
    model = torch.load('lansinuote/model/ractangle.model')
    model.eval()
    x, y = next(iter(loader))

    out = model(x).argmax(dim=1)

    print(out, y)

if __name__ == '__main__':
    # train()
    test()