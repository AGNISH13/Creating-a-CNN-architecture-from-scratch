

import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating our CNN architecture

class my_model(nn.Module):
      def __init__(self):
            super(my_model, self).__init__()

            # Specifying the contents of the layers of our CNN

            self.flatten = nn.Flatten()
            self.conv1 = nn.Conv2d(1, 64, 7, 2)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)

            self.conv1_incp = nn.Conv2d(64, 64, 1, padding='same')
            self.conv2_incp = nn.Conv2d(64, 64, 3, padding='same')
            self.conv3_incp = nn.Conv2d(64, 64, 5, padding='same')
            self.pool_incp = nn.MaxPool2d(3, 1)
            self.pad_incp = nn.ConstantPad2d((1,1,1,1), 0)

            self.conv2 = nn.Conv2d(256, 512, 5, 1, padding='same')
            self.bn2 = nn.BatchNorm2d(512)
            self.conv3 = nn.Conv2d(512, 512, 3, 1, padding='same')

            self.pad1 = nn.ConstantPad2d((2,3,2,3), 0)
            self.pad2 = nn.ConstantPad2d((1,2,1,2), 0)
            self.pad3 = nn.ConstantPad2d((0,1,0,1), 0)

            self.drop1 = nn.Dropout2d(0.3)
            self.drop2 = nn.Dropout2d(0.4)

            self.fc1 = nn.Linear(32768, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 10)
    
      def forward(self, x):


            # Arranging the layers of our CNN

            # Layer 1
            x = self.bn1(F.relu(self.conv1(self.pad1(x))))
            x = self.pool(x)

            # Layer 2 (Inception Block)
            output1 = F.relu(self.conv1_incp(x))
            output2 = F.relu(self.conv2_incp(output1))
            output3 = F.relu(self.conv3_incp(output1))
            output4 = F.relu(self.conv1_incp(self.pool_incp(self.pad_incp(x))))
            x = torch.cat([output1, output2, output3, output4], dim=1)

            # Layer 3
            x = self.pool(x)
            x = self.drop1(x)

            # Layer 4
            x = self.bn2(F.relu(self.conv2(self.pad2(x))))
            x = self.pool(x)
            x = self.drop1(x)

            # Layer 5
            x = self.bn2(F.relu(self.conv3(self.pad3(x))))
            x = self.pool(x)
            x = self.drop1(x)

            # Layer 6
            x = self.bn2(F.relu(self.conv3(x)))
            x = self.drop2(x)

            # Flattening our Tensor end to end
            x = self.flatten(x)

            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.drop1(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x