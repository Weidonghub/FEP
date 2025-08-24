import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import tensor
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import copy
import torch.nn as nn
import torch.optim as optim
import time
import math
import gc
import sys
import bisect
import itertools
from torch import mean, not_equal, ne
# Case 0：HPCC = 0，AE2 = 0; case 1：HPCC = 1，AE2 = 0; case2：HPCC = 1，AE2 = 1;
# Flag for High-Performance Computing Platform (HPCC): 1 = enabled, 0 = disabled
HPCC = 1
# AE2: set to 1 to save in model2, 0 model
AE2 = 0
# Use custom EDPCE loss
use_custom_CE = 1
# Number of bits in each subsequence
k1 = 2
k2 = 4
# Total number of bits in one sequence
k = k1 + k2

# Protection weights
lambda1 = 0.5
lambda2 = 0.5

# Dynamic weighting coefficients
zeta1 = 1
zeta2 = 1
# Penalty factor
loss_pf = 6
# Learning rate 4 ** 10^-5
learning_rate = 0.00004
# Threshold for saving additional model
model_save_threshold = -20 * 0.000001
# Temporary model save interval
temp_save_interval = 50
# Batch size for training
batch_size = 2048*16
# Batch size for testing
batch_size_test = 40960
if HPCC == 1:
    batch_size_test = 1024000
# Training and validation dataset size
num_train_val = batch_size * 600
# Train/validation split ratio
ratio = 0.75
# Number of batches for validation
val_batch_count = 4
# Batch size for validation
batch_size_val = int(num_train_val * ratio / val_batch_count)
# Evaluation: average over three Eb/N0 values (16, 16.5, 17 dB)
eval_ebn0_start = 16
num_ebn0_points = 3
# Initial Eb/N0 and step size for training (in dB)
train_ebn0_start = 0
train_ebn0_step = 0.5
# Save the model only if the loss is below this threshold
save_model_threshold = 0.5
# Training epochs corresponding to each Eb/N0 value (in dB)
epochs_per_ebn0 = [
    # 1-10
    40, 40, 35, 35, 30, 30, 30, 30, 25, 25,
    # 11-20
    25, 25, 20, 20, 20, 20, 15, 15, 15, 15,
    # 21-30
    15, 15, 15, 15, 15, 15, 15, 15, 15, 10,
    # 31-37
    10, 10, 10, 10, 8, 5, 3]
# Cumulative sum of training epochs per Eb/N0
cumulative_epochs = []
running_total = 0
for epoch_count in epochs_per_ebn0:
    running_total += epoch_count
    cumulative_epochs.append(running_total)

# Total number of training epochs
total_epochs = sum(epochs_per_ebn0)
# Models trained beyond this epoch need evaluation
eval_threshold_epoch = cumulative_epochs[32]
# Models trained beyond these epochs will be partially evaluated to save computation time
partial_eval_epoch1 = cumulative_epochs[28]
partial_eval_epoch2 = cumulative_epochs[30]
# On HPCC, fully evaluate models beyond partial_eval_epoch1 and partial_eval_epoch2
num_partial_eval1 = 5
num_partial_eval2 = 5
if HPCC == 1:
    num_partial_eval1 = partial_eval_epoch2 - partial_eval_epoch1
    num_partial_eval2 = eval_threshold_epoch - partial_eval_epoch2
# Number of symbols
num_symbols = 1
# Number of channel uses
num_channel_uses = 1
# Number of bits per channel use
bits_per_channel = k / num_channel_uses
# Interval for printing loss
loss_print_interval = 60
# Theoretical BER values for 64-QAM
qam_64bit = [0.172952528588416, 0.160030733007205, 0.146123740615210, 0.131316831312050,0.115764064135706,
             0.0997041650641596, 0.0834726667298508, 0.0675045750685181,0.0523197982772160, 0.0384829892255838,
             0.0265326098261460, 0.0168835290399689, 0.00972398503997060, 0.00494598719497789, 0.00215400375717990,
             0.000772472180420457, 0.000217173959159420, 4.49888734403963e-05, 6.35114807198657e-06]
# Reference BER values of 64-QAM at Eb/N0 = [16, 16.5, 17] dB (for validation)
qam_64bit_val = [0.000217173959159420, 0.000103150552250524, 4.49888734403963e-05]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate weight vector for binary-to-decimal conversion
def binary_weights(k):
    tmp_array = np.zeros(shape=k)
    for i in range(k):
        tmp_array[i] = 2 ** i
    int_data = tmp_array[::-1]  # [8. 4. 2. 1.]
    int_data = np.reshape(int_data, newshape=(k, 1))
    return int_data


# Convert a decimal integer to a binary list of specified length.
def dec_to_bin_list(value, bit_length):
    bit_length = int(bit_length)
    binary_array = []
    for i in range(bit_length):
        bit = value & (1 << (bit_length - i - 1))
        binary_array.append(1 if bit > 0 else 0)
    return binary_array


# Generate training data subsequences
# s1
torch.manual_seed(1)
s1_train_data = np.random.randint(low=0, high=2, size=(num_train_val, k1 * num_symbols))  # (12800, 40)
s1_train_data = np.reshape(s1_train_data, newshape=(num_train_val, num_symbols, k1))  # (12800, 10, 4)
print('s1 shape:', s1_train_data.shape)
int_data1 = binary_weights(k1)
s1_one_hot_data = np.dot(s1_train_data, int_data1)
s1_one_hot_data = torch.from_numpy(s1_one_hot_data)
s1_vec_one_hot = F.one_hot(s1_one_hot_data.to(torch.int64), 2 ** k1)
s1_vec_one_hot = torch.squeeze(s1_vec_one_hot, dim=2)
s1_vec_one_hot = s1_vec_one_hot.float()
print('s1 one-hot:', s1_vec_one_hot.shape)

# s2
torch.manual_seed(2)
s2_train_data = np.random.randint(low=0, high=2, size=(num_train_val, k2 * num_symbols))  # (12800, 40)
s2_train_data = np.reshape(s2_train_data, newshape=(num_train_val, num_symbols, k2))  # (12800, 10, 4)
print('s2 shape:', s2_train_data.shape)
int_data2 = binary_weights(k2)
s2_one_hot_data = np.dot(s2_train_data, int_data2)
s2_one_hot_data = torch.from_numpy(s2_one_hot_data)
s2_vec_one_hot = F.one_hot(s2_one_hot_data.to(torch.int64), 2 ** k2)
s2_vec_one_hot = torch.squeeze(s2_vec_one_hot, dim=2)
s2_vec_one_hot = s2_vec_one_hot.float()
print('s2 one-hot:', s2_vec_one_hot.shape)

# Split the dataset into training and validation sets
s1_train, s1_val, s2_train, s2_val = train_test_split(s1_vec_one_hot, s2_vec_one_hot, test_size=ratio, random_state=42)
# Obtain the bit subsequences corresponding to the validation set
s1_val_bin = np.argmax(s1_val, axis=2)
tmp1 = s1_val_bin.reshape(-1)
s1_val_bin = torch.tensor([dec_to_bin_list(num.item(), k1) for num in tmp1.flatten()])
# Obtain the bit subsequences corresponding to the validation set
s2_val_bin = np.argmax(s2_val, axis=2)
tmp2 = s2_val_bin.reshape(-1)
s2_val_bin = torch.tensor([dec_to_bin_list(num.item(), k2) for num in tmp2.flatten()])
# Create TensorDataset for training and validation sets
train_dataset = TensorDataset(s1_train, s2_train)
val_dataset = TensorDataset(s1_val, s2_val)
# Create DataLoader for the training set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create DataLoader for the validation set
val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)


# Normalize the input tensor to have unit average power.
def normalization(x):
    a = x ** 2
    b = a[:, 0, :]
    c = a[:, 1, :]
    d = b + c
    k_mean = torch.mean(d, dtype=torch.float32)
    k_mod = torch.sqrt(k_mean)
    return x / k_mod


# Simulate an AWGN (Additive White Gaussian Noise) channel.
def channel_layer(x, sigma):
    w = torch.normal(mean=0.0, std=sigma, size=x.shape)
    w = w.to(device)
    return x + w


# EDPCE loss
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, y_true, target):
        dim = math.log2(y_true.shape[1])
        dim1 = y_true.shape[1]
        dim0 = y_true.shape[0]
        # Take logarithm of the softmax output (adding epsilon to avoid log(0))
        log_output = torch.log(y_true + 1e-8)
        loss_func = nn.NLLLoss()
        target_true = torch.argmax(target, dim=1)
        # Standard CE term
        ce_loss_output = loss_func(log_output, target_true)
        # Calculate the penalty term for the custom loss
        row_values = torch.tensor(list(itertools.product([0, 1], repeat=int(dim))), dtype=torch.int64)
        bit_tensor = row_values.unsqueeze(0).expand(y_true.shape[0], -1, -1)
        target_true1 = torch.tensor(target_true)
        bit_temp = torch.tensor([dec_to_bin_list(num.item(), dim) for num in target_true1.flatten()])
        bit_temp1 = torch.tensor(bit_temp)
        bit_temp1 = bit_temp1.unsqueeze(1).expand(-1, dim1, -1)
        error1 = torch.tensor(bit_tensor != bit_temp1)
        error_num = torch.sum(error1, dim=-1)
        error_num_1 = loss_pf ** error_num
        error_num_2 = torch.where(error_num_1 == 1, 0, error_num_1)
        # Loss value of penalty term
        pt_loss_output = torch.sum(error_num_2 * y_true)/(dim0 * dim1)
        # EDPCE loss value
        custom_loss = ce_loss_output + pt_loss_output
        return custom_loss


# Number of channels
num_channels_s1 = 128
num_channels_s2 = 128


# Neural network architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.s1_input = nn.Sequential(
            nn.Linear(2 ** k1, num_channels_s1),
            nn.ReLU()
        )
        self.s2_input = nn.Sequential(
            nn.Linear(2 ** k2, num_channels_s2),
            nn.ReLU()
        )
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels_s1 + num_channels_s2, 256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 2, kernel_size=1, stride=1),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, num_channels_s1 + num_channels_s2, kernel_size=1, stride=1),
            nn.BatchNorm1d(num_channels_s1 + num_channels_s2),
            nn.ReLU(),
        )
        self.s1_decoded = nn.Sequential(
            nn.Linear(num_channels_s1 + num_channels_s2, num_channels_s1),
            nn.ReLU(),
            nn.Linear(num_channels_s1, 2 ** k1),

            nn.Softmax(dim=-1)
        )
        self.s2_decoded = nn.Sequential(
            nn.Linear(num_channels_s1 + num_channels_s2, num_channels_s2),
            nn.ReLU(),
            nn.Linear(num_channels_s2, 2 ** k2),

            nn.Softmax(dim=-1)
        )

    def forward(self, x, n_sigma):
        x_s1 = x[0]
        x_s2 = x[1]
        x1 = self.s1_input(x_s1)
        x2 = self.s2_input(x_s2)
        x = torch.cat((x1, x2), -1)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = normalization(x)
        x = channel_layer(x, n_sigma)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        x1 = self.s1_decoded(x)
        x2 = self.s2_decoded(x)
        x = [x1, x2]
        return x


# Initialize the Autoencoder model
autoencoder = Autoencoder()
autoencoder = autoencoder.to(device)
# Define the loss function
if use_custom_CE == 1:
    criterion = CustomCrossEntropyLoss()
if use_custom_CE == 0:
    criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# Initialize optimizer (Adam) with learning rate
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
save_num = 0
save_temp_num = 0
sys.stdout.flush()
s_time = time.time()
# Training
for epoch in range(total_epochs):
    print("---------- Start training epoch {}/{} ----------".format(epoch + 1, total_epochs))
    # Set model to training mode
    autoencoder.train()
    # Compute noise standard deviation based on Eb/N0
    stage = bisect.bisect_right(cumulative_epochs, epoch)
    train_Eb_dB = train_ebn0_start + train_ebn0_step * stage
    noise_sigma = np.sqrt(1 / (2 * bits_per_channel * 10 ** (train_Eb_dB / 10)))
    # Counter for printing loss
    num_temp = 0
    for data in train_loader:
        inputs = data
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        output = autoencoder(inputs, noise_sigma)
        # Compute loss for subsequence 1
        output_s1 = output[0].reshape(-1, 2 ** k1)
        output_s1 = output_s1.to(device)
        input_s1 = inputs[0].reshape(-1, 2 ** k1)
        input_s1 = input_s1.to(device)
        loss_s1 = criterion(output_s1, input_s1)
        # Compute loss for subsequence 2
        output_s2 = output[1].reshape(-1, 2 ** k2)
        output_s2 = output_s2.to(device)
        input_s2 = inputs[1].reshape(-1, 2 ** k2)
        input_s2 = input_s2.to(device)
        loss_s2 = criterion(output_s2, input_s2)
        # Compute the composite loss
        zeta1 = loss_s1 / (loss_s1 + loss_s2)
        zeta2 = loss_s2 / (loss_s1 + loss_s2)
        loss = lambda1 * zeta1 * loss_s1 + lambda2 * zeta2 * loss_s2
        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print loss
        num_temp = num_temp + 1
        if (num_temp % loss_print_interval) == 0:
            print("loss:{:.6f}".format(loss.item()))

    # Compute the corresponding bit sequence s1
    input_m1 = np.argmax(inputs[0], axis=2)
    input_m1 = input_m1.reshape(-1)
    input_bin1 = torch.tensor([dec_to_bin_list(num.item(), k1) for num in input_m1.flatten()])
    # Compute the corresponding bit sequence s2
    input_m2 = np.argmax(inputs[1], axis=2)
    input_m2 = input_m2.reshape(-1)
    input_bin2 = torch.tensor([dec_to_bin_list(num.item(), k2) for num in input_m2.flatten()])
    # Generate the bit sequence corresponding to the decoded output
    output_m1 = np.argmax(output[0].detach().numpy(), axis=2)
    output_m1 = output_m1.reshape(-1)
    output_bin1 = torch.tensor([dec_to_bin_list(num.item(), k1) for num in output_m1.flatten()])
    # Generate the bit sequence corresponding to the decoded output
    output_m2 = np.argmax(output[1].detach().numpy(), axis=2)
    output_m2 = output_m2.reshape(-1)
    output_bin2 = torch.tensor([dec_to_bin_list(num.item(), k2) for num in output_m2.flatten()])
    # Calculate the BER between predicted and true bit sequences
    ber_temp1 = torch.tensor(input_bin1 == output_bin1)
    ber_temp1 = ber_temp1.float()
    ber_train1 = torch.mean(ber_temp1)
    # Calculate the BER between predicted and true bit sequences
    ber_temp2 = torch.tensor(input_bin2 == output_bin2)
    ber_temp2 = ber_temp2.float()
    ber_train2 = torch.mean(ber_temp2)

    print("train Eb_dB:{}".format(train_Eb_dB))
    print("test loss:{:.6f}".format(loss.data))
    print("test loss1:{:.6f}, test loss2:{:.6f}".format(loss_s1.data, loss_s2.data))
    print("acc1:{:.6f}, acc2:{:.6f}".format(ber_train1, ber_train2))
    sys.stdout.flush()
    # Temporarily save the model
    if epoch % temp_save_interval == 0:
        print("save temp model branch")
        save_temp_num = save_temp_num + 1
        if HPCC == 0:
            PATH = 'tran_temps{}'.format(save_temp_num)
        if HPCC == 1 and AE2 == 0:
            PATH = 'tran_temps{}'.format(save_temp_num)
        if HPCC == 1 and AE2 == 1:
            PATH = 'tran_temps{}'.format(save_temp_num)
        torch.save(autoencoder, PATH)

    # Validate the model when training performance is good and save the model with the lowest BER
    if eval_threshold_epoch <= epoch or (partial_eval_epoch1 <= epoch < (partial_eval_epoch1 + num_partial_eval1)) or (partial_eval_epoch2 <= epoch < (partial_eval_epoch2 + num_partial_eval2)):
        autoencoder.eval()
        total_loss = 0
        ber_s1 = []
        ber_s2 = []
        ber_s_all = []
        ber_s12_diff = []
        with torch.no_grad():
            for num_i in range(0, num_ebn0_points):
                output0 = []
                output1 = []
                # Noise Sigma at this Eb/N0
                Eb_N0_dB = eval_ebn0_start + num_i * 0.5
                noise_sigma = np.sqrt(1 / (2 * bits_per_channel * 10 ** (Eb_N0_dB / 10)))
                for data in val_loader:
                    inputs = data
                    inputs[0] = inputs[0].to(device)
                    inputs[1] = inputs[1].to(device)
                    output = autoencoder(inputs, noise_sigma)
                    output0.append(output[0])
                    output1.append(output[1])
                # Combine predictions from all batches of the validation set
                output0 = torch.cat(output0, dim=0)
                output1 = torch.cat(output1, dim=0)
                # Generate the bit sequence corresponding to the decoded output
                s1_position = np.argmax(output0, axis=2)
                tmp1 = s1_position.reshape(-1)
                tmp1 = torch.tensor([dec_to_bin_list(num.item(), k1) for num in tmp1.flatten()])
                # Calculate the BER between predicted and true bit sequences
                error_temp_ber1 = torch.tensor(s1_val_bin != tmp1)
                error_temp_ber1 = error_temp_ber1.float()
                bit1_error_rate = torch.mean(error_temp_ber1)
                bit1_error_rate = bit1_error_rate.numpy()
                ber_s1.append(bit1_error_rate)
                # Generate the bit sequence corresponding to the decoded output
                s2_position = np.argmax(output1, axis=2)
                tmp2 = s2_position.reshape(-1)
                tmp2 = torch.tensor([dec_to_bin_list(num.item(), k2) for num in tmp2.flatten()])
                # Calculate the BER between predicted and true bit sequences
                error_temp_ber2 = torch.tensor(s2_val_bin != tmp2)
                error_temp_ber2 = error_temp_ber2.float()
                bit2_error_rate = torch.mean(error_temp_ber2)
                bit2_error_rate = bit2_error_rate.numpy()
                ber_s2.append(bit2_error_rate)
                # Calculate the overall BER
                temp_s12 = (k1 * bit1_error_rate + k2 * bit2_error_rate) / (k1 + k2)
                ber_s_all.append(temp_s12)
                # Compute and store the BER difference between the decoded output and reference 64QAM BER
                ber_s12_diff.append(temp_s12 - qam_64bit_val[num_i])
        # Average BER at the target Eb/N0
        ber_s12_diff_mean = np.mean(ber_s12_diff)
        print(ber_s12_diff_mean)
        # Save the model with the lowest BER
        if ber_s12_diff_mean < save_model_threshold:
            print("Saving the main model")
            save_model_threshold = ber_s12_diff_mean
            if HPCC == 0:
                PATH = 'tran_s'
            if HPCC == 1 and AE2 == 0:
                PATH = 'tran_s'
            if HPCC == 1 and AE2 == 1:
                PATH = 'tran_s'
            torch.save(autoencoder, PATH)
        # Save the model only if its performance (BER) is below the threshold
        if ber_s12_diff_mean < model_save_threshold:
            print("Saving the branch model")
            save_num = save_num + 1
            if HPCC == 0:
                PATH = 'tran_s{}'.format(save_num)
            if HPCC == 1 and AE2 == 0:
                PATH = 'tran_s{}'.format(save_num)
            if HPCC == 1 and AE2 == 1:
                PATH = 'tran_s{}'.format(save_num)
            torch.save(autoencoder, PATH)

    e_time = time.time()
    print(e_time - s_time)
    sys.stdout.flush()

print("model branch num = %d" % save_num)


##########################################################


# Test the trained model's performance
num_test_val = batch_size_test * 100
# Generate testing data subsequences
# s1
torch.manual_seed(1)
s1_test_data = np.random.randint(low=0, high=2, size=(num_test_val, k1 * num_symbols))  # (12800, 40)
s1_test_data = np.reshape(s1_test_data, newshape=(num_test_val, num_symbols, k1))  # (12800, 10, 4)
print('s1 shape:', s1_test_data.shape)
int_data1 = binary_weights(k1)
s1_one_hot_data = np.dot(s1_test_data, int_data1)  # dot做内积运算 (12800, 10, 1)
s1_one_hot_data = torch.from_numpy(s1_one_hot_data)
s1_vec_one_hot = F.one_hot(s1_one_hot_data.to(torch.int64), 2 ** k1)
s1_vec_one_hot = torch.squeeze(s1_vec_one_hot, dim=2)  # torch自带函数删除第二维度
s1_vec_one_hot = s1_vec_one_hot.float()
print('s1 one-hot:', s1_vec_one_hot.shape)

# s2
torch.manual_seed(2)
s2_test_data = np.random.randint(low=0, high=2, size=(num_test_val, k2 * num_symbols))  # (12800, 40)
s2_test_data = np.reshape(s2_test_data, newshape=(num_test_val, num_symbols, k2))  # (12800, 10, 4)
print('s2 shape:', s2_test_data.shape)
int_data2 = binary_weights(k2)
s2_one_hot_data = np.dot(s2_test_data, int_data2)  # dot做内积运算 (12800, 10, 1)
s2_one_hot_data = torch.from_numpy(s2_one_hot_data)
s2_vec_one_hot = F.one_hot(s2_one_hot_data.to(torch.int64), 2 ** k2)
s2_vec_one_hot = torch.squeeze(s2_vec_one_hot, dim=2)  # torch自带函数删除第二维度
s2_vec_one_hot = s2_vec_one_hot.float()
print('s2 one-hot:', s2_vec_one_hot.shape)

len_temp_0 = s1_vec_one_hot.shape[-1]
len_temp_1 = s2_vec_one_hot.shape[-1]

# Create TensorDataset for testing
test_dataset = TensorDataset(s1_vec_one_hot, s2_vec_one_hot)
# Length of the test dataset
test_data_size = len(test_dataset)
# Create DataLoader for the testing set
test_loader = DataLoader(dataset=test_dataset, batch_size=num_test_val * num_symbols, shuffle=False)
# Directory containing models for testing
if HPCC == 0:
    PATH = 'tran_s'
if HPCC == 1 and AE2 == 0:
    PATH = 'tran_s'
if HPCC == 1 and AE2 == 1:
    PATH = 'tran_s'

Vec_Eb_N0 = []
s1_test_ber = []
s2_test_ber = []
s_test_ber = []
s1_test_ser = []
s2_test_ser = []
s_test_ser = []

autoencoder = autoencoder.to(device)
autoencoder.eval()
# Start testing
print('start simulation ...')
start = time.perf_counter()
for Eb_N0_dB in np.arange(10, 18.5, 0.5):
    # Compute noise standard deviation based on Eb/N0
    noise_sigma = np.sqrt(1 / (2 * bits_per_channel * 10 ** (Eb_N0_dB / 10)))
    output_0 = torch.empty(0, num_symbols, len_temp_0)
    output_1 = torch.empty(0, num_symbols, len_temp_1)
    with torch.no_grad():
        for data in test_loader:
            inputs = data
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            output_test = autoencoder(inputs, noise_sigma)
            output_0 = torch.cat((output_test[0], output_0), dim=0)
            output_1 = torch.cat((output_test[1], output_1), dim=0)

    # Compute the symbol error rate (SER) s1
    s1_position = np.argmax(output_0, axis=2)
    tmp1_1 = np.reshape(s1_position, newshape=s1_one_hot_data.shape)  # （6400， 100， 1）
    error_temp_ser1 = torch.tensor(s1_one_hot_data != tmp1_1)
    error_temp_ser1 = error_temp_ser1.float()
    s1_ser = torch.mean(error_temp_ser1)
    # Compute the bit error rate (BER)
    tmp1 = tmp1_1.reshape(-1)
    tmp1 = torch.tensor([dec_to_bin_list(num.item(), k1) for num in tmp1.flatten()])
    tmp1 = tmp1.numpy()
    tmp1 = np.reshape(tmp1, newshape=s1_test_data.shape)
    error_temp_ber1 = torch.tensor(s1_test_data != tmp1)
    error_temp_ber1 = error_temp_ber1.float()
    s1_ber_test = torch.mean(error_temp_ber1)
    # Compute the symbol error rate (SER) s2
    s2_position = np.argmax(output_1, axis=2)
    tmp2_1 = np.reshape(s2_position, newshape=s2_one_hot_data.shape)  # （6400， 100， 1）
    error_temp_ser2 = torch.tensor(s2_one_hot_data != tmp2_1)
    error_temp_ser2 = error_temp_ser2.float()
    s2_ser = torch.mean(error_temp_ser2)
    # Compute the bit error rate (BER)
    tmp2 = tmp2_1.reshape(-1)
    tmp2 = torch.tensor([dec_to_bin_list(num.item(), k2) for num in tmp2.flatten()])
    tmp2 = tmp2.numpy()
    tmp2 = np.reshape(tmp2, newshape=s2_test_data.shape)
    error_temp_ber2 = torch.tensor(s2_test_data != tmp2)
    error_temp_ber2 = error_temp_ber2.float()
    s2_ber_test = torch.mean(error_temp_ber2)

    print('Eb/N0 = ', Eb_N0_dB)
    print("BER1 = %.10f" % s1_ber_test)
    print("BER2 = %.10f" % s2_ber_test)
    print('\n')
    sys.stdout.flush()

    # Save the computed BER and SER
    Vec_Eb_N0.append(Eb_N0_dB)
    s1_test_ser.append(s1_ser)
    s2_test_ser.append(s2_ser)
    s_test_ser.append((s1_ser + s2_ser) / 2)
    s1_test_ber.append(s1_ber_test)
    s2_test_ber.append(s2_ber_test)
    s_test_ber.append((k1 * s1_ber_test + k2 * s2_ber_test) / (k1 + k2))


# Plot the BER figure
marker_size = 4
font_size = 13
fig = plt.figure(num=1,dpi=300, figsize=(10, 6))  # 新建一个图像窗口
plt.semilogy(Vec_Eb_N0, s1_test_ber, marker='*', markersize=marker_size, color='blue', label='s1_ber', linewidth=1)
plt.semilogy(Vec_Eb_N0, s2_test_ber, marker='x', markersize=marker_size, color='blue', label='s2_ber', linewidth=1)
plt.semilogy(Vec_Eb_N0, s_test_ber, color='green', label='s_ber', linewidth=1)
plt.semilogy(range(0, 19), qam_64bit, color='red', label='qam_64 ber', linewidth=1)
plt.legend(loc=0, fontsize=font_size-2, markerscale=1)
plt.yticks(size=font_size)
plt.xticks(size=font_size)
plt.xlabel('Eb/N0', fontsize=font_size)
plt.ylabel('BER', fontsize=font_size)

plt.title(" BER ")
plt.grid('true')

if HPCC == 0:
    plt.savefig('BER.png')
if HPCC == 1 and AE2 == 0:
    plt.savefig('BER.png')
if HPCC == 1 and AE2 == 1:
    plt.savefig('BER2.png')
plt.show()
plt.close()
