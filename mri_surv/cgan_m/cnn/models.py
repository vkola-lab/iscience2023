# Network Models
# Created: 5/21/2021
# Status: OK
# Consider merge into 1 main file
import torch
import torch.nn as nn
import copy
import sys
import matplotlib.pyplot as plt

class _CNN_Pre(nn.Module):
    # The Model that will be used for pre-trained model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 0, bias=False)
        # torch.nn.init.uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        self.conv5 = nn.Conv3d(8*fil_num, 16*fil_num, 3, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm3d(16*fil_num)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        # self.mp = nn.MaxPool3d(5)
        self.mp = nn.MaxPool3d(2)

        self.feature_length = 2*fil_num*28*34*28
        # self.feature_length = 16*fil_num*1*2*1
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        # x = self.dr(self.mp(self.conva(self.bn3(self.conv3(x)))))
        # x = self.dr(self.mp(self.conva(self.bn4(self.conv4(x)))))
        # x = self.dr(self.mp(self.conva(self.bn5(self.conv5(x)))))
        # x = self.mp(self.conva(self.bn1(self.conv1(x))))
        # x = self.mp(self.conva(self.bn2(self.conv2(x))))
        # x = self.mp(self.conva(self.bn3(self.conv3(x))))
        # x = self.mp(self.conva(self.bn4(self.conv4(x))))
        # print(torch.mean(x))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x

class _CNN_Transfer(nn.Module):
    # The Model that will be used for transfer-trained model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 0, bias=False)
        # torch.nn.init.uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 0, bias=False)
        # torch.nn.init.uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 0, bias=False)
        # torch.nn.init.uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 0, bias=False)
        # torch.nn.init.uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm3d(8*fil_num)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2)

        # self.feature_length = 2*fil_num*10*13*9 # for larger input
        self.feature_length = 8*fil_num*7*8*7
        self.feature_length = 8*fil_num*5*7*5
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        # print(torch.mean(x))
        # print(x.shape)
        # x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        # x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        # x = self.dr(self.mp(self.conva(self.bn3(self.conv3(x)))))
        # x = self.dr(self.mp(self.conva(self.bn4(self.conv4(x))))
        x = self.mp(self.conva(self.bn1(self.conv1(x))))
        x = self.mp(self.conva(self.bn2(self.conv2(x))))
        x = self.mp(self.conva(self.bn3(self.conv3(x))))
        x = self.mp(self.conva(self.bn4(self.conv4(x))))
        # print(torch.mean(x))
        # print(x.shape)
        # x = self.dr(self.conva(self.bn3(self.conv3(x))))
        # x = self.dr(self.conva(self.bn4(self.conv4(x))))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x

class _CNN_Surv_Res(nn.Module):
    # The Model that will be used for base model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        # ResNet introduction!!!!!!!!!!
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        self.conv5 = nn.Conv3d(8*fil_num, 16*fil_num, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm3d(16*fil_num)
        # torch.nn.init.uniform_(self.conv4.weight)

        self.res_block1 = nn.Conv3d(in_channels, 4*fil_num, 1, 4, 1, bias=False)
        self.res_block2 = nn.Conv3d(4*fil_num, 16*fil_num, 1, 4, 1, bias=False)
        # self.res_block3 = nn.Conv3d(8*fil_num, 16*fil_num, 1, 2, 0, bias=False)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2, padding=1)

        # self.feature_length = 16*fil_num*1*2*1
        self.feature_length = 16*fil_num*5*6*5
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        res = x
        # print('1')
        # print(x.shape)
        x = self.conva(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        # print('2')
        x = self.conva(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        res = self.res_block1(res)
        # print('res', res.shape)
        # print('3')
        x = self.conva(self.bn3(self.conv3(x)+res))
        # x = self.conva(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        # print('4')
        x = self.conva(self.bn4(self.conv4(x)))
        # print(x.shape)
        x = self.dr(self.mp(x))
        # print(x.shape)
        res = self.res_block2(res)
        # print('res', res.shape)
        # print('5')
        x = self.conva(self.bn5(self.conv5(x)+res))
        # x = self.conva(self.bn5(self.conv5(x)))
        x = self.dr(self.mp(x))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x

class _CNN(nn.Module):
    # The Model that will be used for pre-trained model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        # torch.nn.init.uniform_(self.conv4.weight)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2)

        self.feature_length = 4*fil_num*13*16*13
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        # print(x.shape)
        x = self.dr(self.mp(self.conva(self.bn3(self.conv3(x)))))
        # x = self.dr(self.mp(self.conva(self.bn4(self.conv4(x)))))
        # x = self.mp(self.conva(self.bn1(self.conv1(x))))
        # x = self.mp(self.conva(self.bn2(self.conv2(x))))
        # x = self.mp(self.conva(self.bn3(self.conv3(x))))
        # x = self.mp(self.conva(self.bn4(self.conv4(x))))
        # print(torch.mean(x))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        x = self.l2(x)
        # x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x

class _CNN_Surv(nn.Module):
    # The Model that will be used for base model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        # Bias?
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 0, bias=True)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 0, bias=True)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 0, bias=True)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 0, bias=True)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        self.conv5 = nn.Conv3d(8*fil_num, 16*fil_num, 3, 1, 0, bias=True)
        self.bn5 = nn.BatchNorm3d(16*fil_num)
        # torch.nn.init.uniform_(self.conv4.weight)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2)

        self.feature_length = 2*fil_num*28*34*28
        # self.feature_length = 4*fil_num*13*16*13
        # self.feature_length = 8*fil_num*5*7*5
        # self.feature_length = 16*fil_num*1*2*1
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, x, stage='train'):
        # print(x.shape)
        x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        # print(x.shape)
        x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        # print(x.shape)
        # x2 = self.dr(self.conva(self.bn3(self.conv3(x))))
        # x = self.dr(self.mp(self.conva(self.bn3(self.conv3(x)))))
        # print('hi')
        # print(x.shape)
        # x = self.dr(self.mp(self.conva(self.bn4(self.conv4(x)))))
        # print(x.shape)
        # x = self.dr(self.mp(self.conva(self.bn5(self.conv5(x)))))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x = self.l2a(self.l2(x))
        # x = self.ac(x)
        return x

class _CNN_Surv_Append(nn.Module):
    # The Model that will be used for base model
    def __init__(self, drop_rate, fil_num=64, in_channels=1, out_channels=1):
        super().__init__()
        # Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        # Bias?
        self.conv1 = nn.Conv3d(in_channels, fil_num, 3, 1, 0, bias=True)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 2*fil_num, 3, 1, 0, bias=True)
        self.bn2 = nn.BatchNorm3d(2*fil_num)
        self.conv3 = nn.Conv3d(2*fil_num, 4*fil_num, 3, 1, 0, bias=True)
        self.bn3 = nn.BatchNorm3d(4*fil_num)
        self.conv4 = nn.Conv3d(4*fil_num, 8*fil_num, 3, 1, 0, bias=True)
        self.bn4 = nn.BatchNorm3d(8*fil_num)
        self.conv5 = nn.Conv3d(8*fil_num, 16*fil_num, 3, 1, 0, bias=True)
        self.bn5 = nn.BatchNorm3d(16*fil_num)
        # torch.nn.init.uniform_(self.conv4.weight)

        # (D−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        # self.conv5 = nn.ConvTranspose3d(fil_num, in_channels, 4, 2, 0, bias=False, output_padding=(1,1,1))
        # self.bn5 = nn.BatchNorm3d(in_channels)

        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        self.mp = nn.MaxPool3d(2)

        self.feature_length = 2*fil_num*28*34*28
        # self.feature_length = 4*fil_num*13*16*13
        # self.feature_length = 8*fil_num*5*7*5
        # self.feature_length = 16*fil_num*1*2*1
        self.l1 = nn.Linear(self.feature_length, 50)
        self.l2 = nn.Linear(50+4, out_channels)
        self.l1a = nn.LeakyReLU()
        self.l2a = nn.Sigmoid()
        # self.l2a = nn.Tanh()
        '''
        RELU & Softmax: not learning
        Sigmoid & LeakyReLU & GELU: weak learning
        '''
        # self.ac = nn.GELU()

        self.num = fil_num

    def forward(self, xs, stage='train'):
        x = xs[0]
        x_a = torch.nan_to_num(xs[1], 0)
        # print(x.shape)
        x = self.dr(self.mp(self.conva(self.bn1(self.conv1(x)))))
        # print(x.shape)
        x = self.dr(self.mp(self.conva(self.bn2(self.conv2(x)))))
        # print(x.shape)
        # x2 = self.dr(self.conva(self.bn3(self.conv3(x))))
        # x = self.dr(self.mp(self.conva(self.bn3(self.conv3(x)))))
        # print('hi')
        # print(x.shape)
        # x = self.dr(self.mp(self.conva(self.bn4(self.conv4(x)))))
        # print(x.shape)
        # x = self.dr(self.mp(self.conva(self.bn5(self.conv5(x)))))
        # print(x.shape)
        # sys.exit()

        x = x.view(-1, self.feature_length)
        x = self.l1a(self.l1(x))
        # x = self.l2(x)
        x_a = torch.reshape(x_a, [x.shape[0], -1])
        x = torch.cat((x,x_a), 1).float()
        x = self.l2a(self.l2(x))
        if torch.isnan(x).any():
            print('problem')
            print(x)
            sys.exit()
        # x = self.ac(x)
        return x


if __name__ == "__main__":
    print('models.py')

    #  use gpu if available
    batch_size = 512
    epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    encoder = _Encoder(in_size=784, drop_rate=.5, out=128, fil_num=128).to(device)
    decoder = _Decoder(in_size=128, drop_rate=.5, out=784, fil_num=128).to(device)
    # model = AE(input_shape=784).to(device)


    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    import torch.optim as optim
    import torchvision

    optimizerE = optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizerD = optim.Adam(decoder.parameters(), lr=learning_rate)

    # mean-squared error loss
    criterion = nn.MSELoss()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss_imp = 0.0
    loss_tot = 0.0
    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizerE.zero_grad()
            optimizerD.zero_grad()

            # compute reconstructions
            # outputs = model(batch_features)
            # print(batch_features.shape)
            # sys.exit()
            vector = encoder(batch_features)
            outputs = decoder(vector)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizerE.step()
            optimizerD.step()

            vector = encoder(batch_features)
            outputs = decoder(vector)
            loss2 = criterion(outputs, batch_features)
            if loss2 < train_loss:
                loss_imp += 1
            loss_tot += 1

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss), 'loss improved: %.2f' % (loss_imp/loss_tot))

    test_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            test_examples = batch_features.view(-1, 784).to(device)
            reconstruction = decoder(encoder(test_examples))
            break
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig("AE.png")
