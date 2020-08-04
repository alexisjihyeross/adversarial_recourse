import torch

# define model class
class SmallModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size = 100):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, 1)
        self.drop = torch.nn.Dropout(p=0.3)
        self.activ = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.drop(self.activ(self.linear1(x)))
        y = self.drop(self.activ(self.linear2(y)))
        y = (self.linear3(y))
        return self.sigmoid(y)