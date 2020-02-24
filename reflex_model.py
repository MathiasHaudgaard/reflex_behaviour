import torch
import torch.nn as nn


class Reflex_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, input):
        input = input/127.5-1.0
        input = self.elu(self.conv_0(input))
        input = self.elu(self.conv_1(input))
        input = self.elu(self.conv_2(input))
        input = self.elu(self.conv_3(input))
        input = self.elu(self.conv_4(input))
        input = self.dropout(input)

        input = input.flatten()
        input = self.elu(self.fc0(input))
        input = self.elu(self.fc1(input))
        input = self.elu(self.fc2(input))
        input = self.fc3(input)

        return input


def main():

    print('Hello World!')

if __name__ == '__main__':

    main()
