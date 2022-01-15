import torch                            #import torch: 使用pytorch框架
import torch.nn as nn                   #import torch.nn as nn: 使用neural network模塊,所有網路的基本類別
from torch.autograd import Variable     #from torch.autograd import Variable： variable像一個容器,可以容納tensor在裡面計算.
import torch.utils.data as Data         #import torch.utils.data as Data: 隨機抽取data的工具,隨機mini-batch
import torchvision                      #import torchvision: 用來生成圖片影片的數據集,流行的pretrained model

EPOCH = 5                #全部data訓練10次
BATCH_SIZE = 50           #每次訓練隨機丟50張圖像進去
LR =0.001                  #learning rate
DOWNLOAD_MNIST = True     #第一次用要先下載data,所以是True
if_use_gpu = 1            #使用gpu

#training data
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    #把灰階從0~255壓縮到0~1
    download=DOWNLOAD_MNIST
)

print(train_data.train_data.size())
print(train_data.train_labels.size())

train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle=True)
#shuffle是隨機從data裡讀去資料.

#testing_data
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
    )

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).float(), requires_grad=False)
#requires_grad = False 不參與反向傳播, testdata不用做

test_y = test_data.test_labels

#神經網路

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 我們開始定義一系列網路如下：  #train data ＝ (1,28,28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                # input channel(EX:RGB)
                out_channels = 16,
                # output feature maps
                kernel_size = 5,
                # filter大小
                stride = 1,
                # 每次convolution移動多少
                padding = 2
                # 在圖片旁邊補0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        # 以上為一層conv + ReLu + maxpool
        # 快速寫法：
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)

        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

    # forward流程:
    # x = x.view(x.size(0), -1) 展平data

cnn = CNN()
if if_use_gpu:
    cnn = cnn.cuda()

#優化器使用Adam
#loss_func 使用CrossEntropy（classification task）
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x, requires_grad=False)
        b_y = Variable(y, requires_grad=False)
        # 決定跑幾個epoch,enumerate把load進來的data列出來成（x,y）

        if if_use_gpu:
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        # 使用cuda加速
        output = cnn(b_x)  # 把data丟進網路中
        loss = loss_function(output, b_y)
        optimizer.zero_grad()  # 計算loss,初始梯度
        loss.backward()  # 反向傳播
        optimizer.step()

        if step % 50 == 0:
            print('Epoch:', epoch, '|step:', step, '|train loss:%.4f' % loss.data)

        # 每100steps輸出一次train loss