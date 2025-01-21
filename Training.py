import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import Model
from Model import DDPM
import pickle

# Select the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 18
number_of_time_steps = 1000

# Values of Beta, Alpha, Alpha_Ber
beta = torch.linspace(1e-4,0.02,number_of_time_steps).to(device)
alpha = (1-beta).to(device)
alpha_ber = torch.cumprod(alpha, dim=0).to(device)

# To Detect Any Problem at the Time of Training
torch.autograd.set_detect_anomaly(True)

# Transformation 
transform_1 = transforms.ToTensor()
transform_2 = transforms.Normalize((0.5),(0.5))

# Data Set and Data Loader
mnist_data = datasets.MNIST(root='/home/idrbt/Desktop/DIFFUSION/Data', train=True, download=True, transform=transform_1)
data_loader = DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True,drop_last=True,num_workers=4)

# Initiate object of DDPM
model = DDPM(input_image_channel=1,output_image_channel=1)

# Keep the Model in Training Mode
model.train()
model.to(device)

# Selection of optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# Selection of Loss
criterion = nn.MSELoss()

loss_each_epoch = []
for i in range(num_epochs):
    total_loss = 0
    for batch_number,(x,l) in enumerate(data_loader):

        x = x.to(device)

        # Padding the Input Image to Change the 
        # Size from 28 X 28 to 32 X 32
        x = F.pad(x,(2,2,2,2))

        # Normalize the Input Image Values from -1 to +1
        x = transform_2(x)

        # Generate the epsilon from White Gaussian Noise
        epsilon = torch.randn(x.shape).to(device)

        # Here, t is a 1 D tensor of Dimension batch_size
        t = torch.randint(0,999,(batch_size,)).to(device)

        # Generate the alpha_ber information
        alpha_ber_required = alpha_ber[t]

        # Reshape alpha_ber to a 4 D Tensor
        alpha_ber_required = alpha_ber_required.reshape(x.shape[0],1,1,1)

        # calculate the noisy version of Images
        # from Original Images
        a = torch.sqrt(alpha_ber_required)

        b = torch.sqrt(1-alpha_ber_required)

        x_t = (a*x) + (b*epsilon)

        # Remove the Previous Gradients
        optimizer.zero_grad()

        # Calculate output of DDPM model
        output = model(x_t, t)

        # Calculate the Loss
        loss = criterion(output,epsilon)
        
        # Calculate the Total Loss
        total_loss = total_loss + loss.item()
        
        # Backward pass
        loss.backward()
        
        # Updation of weight
        optimizer.step()

        # Display important information of training
        print('batch Number =', batch_number+1)
        print('Epoch Number =', i+1)
        print('Loss is =',loss.item())
    
    # Store the average loss value of each epoch
    loss_each_epoch.append((total_loss)/(batch_number+1))

# Save the DDPM model
torch.save(model.state_dict(),'/home/idrbt/Desktop/DIFFUSION/DDPM_MODEL_18_Epochs.pth')

# Save the loss values of all epochs in a file
f = open('Loss_values_18.txt','wb')
pickle.dump(loss_each_epoch,f)
f.close()



