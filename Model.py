import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Select the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()


# Implementation of Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self,time_steps=1000,embedding_dim=512):
        # According to Paper, time_steps is 1000
        # It is the total number of time steps we used
        # to convert original image to a complete white gaussian noise

        # We have selected embedding_dim as 512, because 512 is the
        # maximum number of channels in our model.
        # We are adding this Positional Encoding information to the input tensor
        # in the channel dimension in the Residual Block. Hence, according to the
        # number of channels of the input tensor, we will truncating the Positional Encoding
        # information in the channel dimension. This truncated Positional Encoding information
        # will be added to the input tensor in the Residual Block.

        super().__init__()
        pe = torch.zeros(time_steps,embedding_dim,device=device)

        even_i = torch.arange(0,embedding_dim,2,device=device)
        odd_i = torch.arange(1,embedding_dim,2,device=device)

        even_denominator = (10000)**(even_i/embedding_dim)
        odd_denominator = (10000)**((odd_i-1)/embedding_dim)

        position = torch.arange(0,time_steps,1,device=device)
        position = position.reshape(time_steps,1)

        pe[:,0:embedding_dim:2] = torch.sin(position/even_denominator)
        pe[:,1:embedding_dim:2] = torch.cos(position/odd_denominator)
        
        self.positional_encoding = pe
        # positional_encoding is a 2D Tensor
        # (time_steps,embedding_dim)
        # (1000 X 512)
        self.embedding_dim = embedding_dim

    def forward(self,t):
        # Here, t is 1 D Tensor of Dimension batch_size
        batch_size = t.shape[0]
        channel_number = self.embedding_dim
        # Above Channel Number is the Maximum Channel Number = 512
        # Selecting only batch_size number of rows from positional_encoding
        # Selection is done by indexing operation through t Tensor
        selected_PE = self.positional_encoding[t]
        # selected_PE is a 2D Tensor
        # (batch_size X embedding_dim)
        # (batch_size X 512)
        selected_PE = selected_PE.reshape(batch_size,channel_number,1,1)
        # Now, selected_PE is a 4D Tensor
        # (batch_size X embedding_dim X 1 X 1)
        # (batch_size X channel_number X 1 X 1)
        # (batch_size X 512 X 1 X 1)
        return selected_PE


# Implementation of Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups, drop_prob = 0.1):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(num_groups=num_groups,num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.dropout = nn.Dropout(p = drop_prob)
        self.group_norm_2 = nn.GroupNorm(num_groups=num_groups,num_channels=channels)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    
    def forward(self, x, position_information):
        # x is 4D input Tensor (batch_size, num_channel, height, width)
        # num_channel of input x varies from resolution to resolution
        # position_information is a 4D Tensor (batch_size X 512 X 1 X 1)
        # We are truncating the position_information in num_channel dimension
        position_information = position_information[:,:x.shape[1],:,:]
        # After truncation we can add input Tensor with position_information
        x = x + position_information
        r = self.group_norm_1(x)
        r = self.relu(r)
        r = self.conv1(r)
        r = self.dropout(r)
        r = self.group_norm_2(r)
        r = self.relu(r)
        r = self.conv2(r)
        # Application of Skip Connection
        x1 = x + r
        return x1


# Implementation of Multihead Attention
# Input is a 4D Tensor (batch_size, num_channel, height, width)
# Output is a 4D Tensor (batch_size, num_channel, height, width)
# Calculation of Scaled Dot Product Attention
def Scaled_Dot_Product_Attention(q, k, v):
    d_k = q.shape[-1]
    scaled_dot_product = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
    attention = F.softmax(scaled_dot_product,dim =-1)
    out = torch.matmul(attention, v)
    return attention, out


class MultiheadAttention(nn.Module):
    def __init__(self, channel, num_heads):
        super().__init__()
        self.channel = channel
        self.num_heads = num_heads
        self.head_dim = channel//num_heads
        self.q_layer = nn.Linear(channel, channel, bias = False)
        self.k_layer = nn.Linear(channel, channel, bias = False)
        self.v_layer = nn.Linear(channel, channel, bias = False)
        self.linear_layer = nn.Linear(channel, channel, bias = False)
    
    def forward(self, x):

        # Size of input tensor is batch_size, channel, height, width
        (b,c,h,w) = x.shape
        # size of x1 is batch_size, channel, height*width
        x1 = x.reshape(b, c, h*w)
        # size of x2 is batch_size, height*width, channel
        x2 = x1.permute(0, 2, 1)

        # Generation of q Tensor
        q = self.q_layer(x2)
        # Here size of q is batch_size, height*width, channel
        q = q.reshape(b, h*w, self.num_heads, self.head_dim)
        # Here size of q is batch_size, height*width, num_heads, head_dim
        q = q.permute(0, 2, 1, 3)
        # Here size of q is batch_size, num_heads, height*width, head_dim

        # Generation of k Tensor
        k = self.k_layer(x2)
        # Here size of k is batch_size, height*width, channel
        k = k.reshape(b, h*w, self.num_heads, self.head_dim)
        # Here size of k is batch_size, height*width, num_heads, head_dim
        k = k.permute(0, 2, 1, 3)
        # Here size of k is batch_size, num_heads, height*width, head_dim

        # Generation of v Tensor
        v = self.v_layer(x2)
        # Here size of v is batch_size, height*width, channel
        v = v.reshape(b, h*w, self.num_heads, self.head_dim)
        # Here size of v is batch_size, height*width, num_heads, head_dim
        v = v.permute(0, 2, 1, 3)
        # Here size of v is batch_size, num_heads, height*width, head_dim

        # Calculation of Multi Head Attention
        (attention, attention_head)= Scaled_Dot_Product_Attention(q, k, v)
        # size of Attention Probability is batch_size, num_heads, height*width, height*width
        # size of Attention Head is batch_size, num_heads, height*width, head_dim

        # Concatination of Multiple Heads
        attention_head = attention_head.permute(0, 2, 1, 3)
        # Here size of Attention Head is batch_size, height*width, num_heads, head_dim)
        attention_head = attention_head.reshape(b, h*w, self.num_heads*self.head_dim)
        # Here size of Attention Head is batch_size, height*width, channel

        # Inter Communication between Multiple Heads
        z = self.linear_layer(attention_head)
        # Here size of z tensor is batch_size, height*width, channel
        z1 = z.permute(0,2,1)
        # Here size of z1 tensor is batch_size, channel, height*width
        z2 = z1.reshape(b,c,h,w)
        # Here size of z2 is batch_size, channel, height, width
        return z2


class Resolution_32_32_Left(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=64,num_groups=32)
        self.rb2 = ResidualBlock(channels=64,num_groups=32)
        self.conv = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
    
    def forward(self, x, pc):
        # Here, x is input tensor and pc is positional coding information
        x = self.rb1(x, pc)
        x = self.rb2(x, pc)
        y = self.conv(x)
        return x,y


class Resolution_16_16_Left(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=128,num_groups=32)
        self.attention_layer = MultiheadAttention(channel=128,num_heads=8)
        self.rb2 = ResidualBlock(channels=128,num_groups=32)
        self.conv = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
    
    def forward(self, x, pc):
        # Here, x is input tensor and pc is positional coding information
        x = self.rb1(x, pc)
        x = self.attention_layer(x)
        x = self.rb2(x, pc)
        y = self.conv(x)
        return x,y


class Resolution_8_8_Left(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=256,num_groups=32)
        self.rb2 = ResidualBlock(channels=256,num_groups=32)
        self.conv = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)
    
    def forward(self, x, pc):
        # Here, x is input tensor and pc is positional coding information
        x = self.rb1(x, pc)
        x = self.rb2(x, pc)
        y = self.conv(x)
        return x,y


class Resolution_4_4_BottleNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=512,num_groups=32)
        self.rb2 = ResidualBlock(channels=512,num_groups=32)
        self.convt = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1)
    
    def forward(self, x, pc):
        # Here, x is input tensor and pc is positional coding information
        x = self.rb1(x, pc)
        x = self.rb2(x, pc)
        y = self.convt(x)
        return y

class Resolution_8_8_Right(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=512,num_groups=32)
        self.rb2 = ResidualBlock(channels=512,num_groups=32)
        self.convt = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1)
    
    def forward(self, x1, x2, pc):
        # x1 is upsample input tensor
        # x2 is residual input tensor from resolution 8 X 8 left
        # pc is positional coding information
        x = torch.cat([x1,x2],dim=1)
        x = self.rb1(x, pc)
        x = self.rb2(x, pc)
        y = self.convt(x)
        return y

class Resolution_16_16_Right(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=384,num_groups=32)
        self.attention_layer = MultiheadAttention(channel=384,num_heads=8)
        self.rb2 = ResidualBlock(channels=384,num_groups=32)
        self.convt = nn.ConvTranspose2d(in_channels=384,out_channels=192,kernel_size=4,stride=2,padding=1)
    
    def forward(self, x1, x2, pc):
        # x1 is upsample input tensor
        # x2 is residual input tensor from resolution 16 X 16 left
        # pc is positional coding information
        x = torch.cat([x1,x2],dim=1)
        x = self.rb1(x, pc)
        x = self.attention_layer(x)
        x = self.rb2(x, pc)
        y = self.convt(x)
        return y

class Resolution_32_32_Right(nn.Module):
    def __init__(self):
        super().__init__()
        self.rb1 = ResidualBlock(channels=256,num_groups=32)
        self.rb2 = ResidualBlock(channels=256,num_groups=32)
        self.conv = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,stride=1,padding=1)
        
    
    def forward(self, x1, x2, pc):
        # x1 is upsample input tensor
        # x2 is residual input tensor from resolution 32 X 32 left
        # pc is positional coding information
        x = torch.cat([x1,x2],dim=1)
        x = self.rb1(x, pc)
        x = self.rb2(x, pc)
        y = self.conv(x)
        return y


class DDPM(nn.Module):
    def __init__(self,input_image_channel, output_image_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_image_channel,64,kernel_size=3,stride=1,padding=1)
        self.poen = PositionalEncoding()
        self.sec1 = Resolution_32_32_Left()
        self.sec2 = Resolution_16_16_Left()
        self.sec3 = Resolution_8_8_Left()
        self.sec4 = Resolution_4_4_BottleNeck()
        self.sec5 = Resolution_8_8_Right()
        self.sec6 = Resolution_16_16_Right()
        self.sec7 = Resolution_32_32_Right()
        self.conv2 = nn.Conv2d(64,output_image_channel,kernel_size=3,stride=1,padding=1)
    
    def forward(self, x, t):
        x = self.conv1(x)
        po_enc = self.poen(t)
        resi_out_1,downcon_out_1 = self.sec1(x,po_enc)
        resi_out_2,downcon_out_2 = self.sec2(downcon_out_1,po_enc)
        resi_out_3,downcon_out_3 = self.sec3(downcon_out_2,po_enc)
        upcon_out_1 = self.sec4(downcon_out_3,po_enc)
        upcon_out_2 = self.sec5(upcon_out_1, resi_out_3, po_enc)
        upcon_out_3 = self.sec6(upcon_out_2, resi_out_2, po_enc)
        pre_output = self.sec7(upcon_out_3, resi_out_1, po_enc)
        output = self.conv2(pre_output)
        return output



