### Add necessary imports ###
from torch import nn
import torch.nn.functional as F 
import torch 

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

### Complete the CNN class here ###
    
class res_blk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(res_blk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) \
                        if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)  
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out)
    
class dense_blk(nn.Module):
    def __init__(self, in_channels, num_layers=3, growth_rate=12, drop_rate=0.2):
        super(dense_blk, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            conv = nn.Conv2d(layer_in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
            bn = nn.BatchNorm2d(growth_rate)
            drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
            self.layers.append(nn.ModuleDict({"conv": conv, "bn": bn, "drop": drop}))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            cat = torch.cat(features, dim=1)
            out = F.relu(layer["bn"](layer["conv"](cat)))
            out = layer["drop"](out)
            features.append(out)
        return torch.cat(features, dim=1)  

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5): # Add arguments and modify as needed
        super(CNN, self).__init__()
        # initial conv layer 
        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        # res_blk 
        self.res1 = res_blk(24, 48)
        self.pool1 = nn.MaxPool2d(2,2)  
        # dense_blk 
        self.dense = dense_blk(in_channels=48, num_layers=3, growth_rate=12, drop_rate=0.2)
        self.pool2 = nn.MaxPool2d(2,2) 

        dense_out = 48 + 3*12  # 84
        self.transition_bn = nn.BatchNorm2d(dense_out)
        self.transition_conv = nn.Conv2d(dense_out, 96, kernel_size=1, bias=False)

        self.final_res1 = res_blk(96, 96)   
        self.final_res2 = res_blk(96, 128)  

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(128, 64), nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout_rate), nn.Linear(64, 32), nn.ReLU(inplace=True),
                                        nn.Dropout(p=dropout_rate), nn.Linear(32, num_classes))
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        x = self.res1(x)                      
        x = self.pool1(x)                     
        x = self.dense(x)                     
        x = self.pool2(x)                     
        x = F.relu(self.transition_bn(x))
        x = self.transition_conv(x)           
        x = self.final_res1(x)                
        x = self.final_res2(x)               
        x = self.global_avg_pool(x)          
        x = x.view(x.size(0), -1)            
        x = self.classifier(x)                
        return x 