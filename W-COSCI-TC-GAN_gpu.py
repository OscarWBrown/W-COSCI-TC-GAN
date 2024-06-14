import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
import sklearn
from matplotlib import pylab as plt
import warnings
import os
import csv
import sys
import traceback
from tqdm import tqdm
from torch.utils.data import Dataset
from matplotlib.lines import Line2D
from torch.autograd import Variable
import seaborn as sns
import scipy.stats as stats
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import STL

file_paths = [
    'GDX', 'NUGT',
    'GDXJ', 'JNUG',
    'IWM', 'RWM',
    'SOXL', 'SOXS',
    'SPXL', 'UPRO',
    'SPY', 'SSO',
    'TQQQ', 'QQQ',
    'TSLL', 'TNA'
]

file_pairs = [
    ('GDX', 'NUGT'),
    ('GDXJ', 'JNUG'),
    ('IWM', 'RWM'),
    ('SOXL', 'SOXS'),
    ('SPXL', 'UPRO'),
    ('SPY', 'SSO'),
    ('TQQQ', 'QQQ'),
    ('TSLL', 'TNA')
]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# parser = argparse.ArgumentParser(description="Run GAN model with specific alpha values")
# args = parser.parse_args()   

with tf.device('/gpu:0'):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_path = '/Users/oscar/Desktop/Quant study/MCD - Project 1/Scripts/X-COSCI-TC-GAN/Full_data/'
    
    
    # FUNCTIONS
    class Loader32(Dataset):
        def __init__(self, data1, data2, length):
            # Ensure both data1 and data2 have at least 'length' number of elements
            assert len(data1) >= length and len(data2) >= length
            self.data1 = data1
            self.data2 = data2
            self.length = length
        
        def __getitem__(self, idx):
            # Extract a window of 'length' elements starting from 'idx' for data1
            data1_window = torch.tensor(self.data1[idx:idx+self.length]).reshape(1, self.length).to(torch.float32)
            
            # Extract a window of 'length' elements starting from 'idx' for data2
            data2_window = torch.tensor(self.data2[idx:idx+self.length]).reshape(1, self.length).to(torch.float32)
            
            # Concatenate the two windows along the first dimension and return
            return torch.cat((data1_window, data2_window), dim=0)
            
        def __len__(self):
            # Calculate the number of possible windows for each data series
            return max(len(self.data1) - self.length, len(self.data2) - self.length, 0)
     
    def clean_data(df):
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        df = df.dropna()  # Drop rows with NaN
        return df

    def check_for_nan(data, name):
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"NaN or Inf found in {name}")
        else:
            print(f"No NaN or Inf found in {name}")

    def reverse_normalization(normalized_df, normalization_values):
        original_data = normalized_df.copy()
        for column in normalized_df.columns:
            # Use the base column name to get the normalization values
            base_column = column.split('_Returns')[0] + '_Returns'
            min_val = normalization_values[base_column]['min']
            max_val = normalization_values[base_column]['max']
            if max_val > min_val:
                original_data[column] = normalized_df[column] * (max_val - min_val) + min_val
        return original_data

    def weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    # Set to 0 for no penalty
    def gradient_penalty(discriminator, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, device=device)
        alpha = alpha.expand(real_data.size())
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def reverse_returns(returns_df, initial_close):
        close_prices = returns_df.copy().astype(float)  # Ensure the dtype is compatible
        close_prices.iloc[0] = initial_close * (1 + close_prices.iloc[0])
        for i in range(1, len(close_prices)):
            close_prices.iloc[i] = close_prices.iloc[i - 1] * (1 + close_prices.iloc[i])
        return close_prices

    def adjust_mean(synthetic_returns, real_mean):
        synthetic_returns -= synthetic_returns.mean(axis=1, keepdims=True)
        synthetic_returns += real_mean
        return synthetic_returns

    def load_generators(data_pairs, seq_len, nz, device, save_dir, num_epochs):
        generators = []
        for I, (df_combined, name1, name2) in enumerate(data_pairs):
            generator = Generator(seq_len, input_dim=nz, output_dim=2).to(device)
            generator.load_state_dict(torch.load(os.path.join(save_dir, f'trained_generator_pair_{name1}_{name2}_epoch_{num_epochs}.pth')))
            generator.eval()
            generators.append(generator)
        return generators

    def generate_synthetic_data(generators, batch_size, nz, seq_len, device):
        synthetic_data_list = []
        for generator in generators:
            noise = torch.randn(batch_size, nz, seq_len).to(device)
            synthetic_data = generator(noise).cpu().detach().numpy()
            synthetic_data_list.append(synthetic_data)
        return synthetic_data_list

    def calc_skewness_kurtosis(data):
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        return skewness, kurtosis

    def calc_autocorrelation(data, lag):
        return pd.Series(data).autocorr(lag=lag)

    def calc_rolling_volatility(data, window):
        return pd.Series(data).rolling(window=window).std()

    def adjust_length(data, length):
        if len(data) > length:
            return data[:length]
        else:
            return np.pad(data, (0, length - len(data)), 'constant', constant_values=np.nan)

    def calculate_fid(mu1, sigma1, mu2, sigma2):
        """Calculate the Frechet Inception Distance (FID) between two distributions."""
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1 @ sigma2)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def compute_statistics(data):
        mu = np.mean(data, axis=0)
        sigma = np.cov(data, rowvar=False)
        if sigma.ndim == 0:  # if the covariance matrix is a scalar
            sigma = np.array([[sigma]])  # convert it to a 2D array
        return mu, sigma

    def calculate_coverage_probability(historical_data, synthetic_data):
        lower_bound = np.min(historical_data)
        upper_bound = np.max(historical_data)
        coverage = np.mean((synthetic_data >= lower_bound) & (synthetic_data <= upper_bound))
        return coverage

    def calculate_cross_correlation(series1, series2):
        return np.correlate(series1 - np.mean(series1), series2 - np.mean(series2), mode='full') / (np.std(series1) * np.std(series2) * len(series1))

    def decompose_time_series(data, period):
        stl = STL(data, period=period, seasonal=13)
        result = stl.fit()
        return result.trend, result.seasonal, result.resid
    
    # PREPROCESSING
    # HISTORICAL CLOSE DATA
    all_data = []
    initial_close_prices = {}  # Dictionary to store initial close prices for each asset

    for file_path in file_paths:
        data = pd.read_parquet(base_path + file_path + '.parquet', engine='pyarrow')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.drop_duplicates(subset='datetime')

        # Filter out data between 4:00 PM and 9:30 AM
        data = data[(data['datetime'].dt.hour > 9) | (data['datetime'].dt.hour == 9) & (data['datetime'].dt.minute >= 30)]
        data = data[data['datetime'].dt.hour < 16]

        data = data.set_index('datetime')[['close']]
        
        # Store the initial close price
        initial_close_prices[file_path] = data['close'].iloc[0]
        
        data.columns = [file_path + '_Close']
        all_data.append(data)

    # Create a common datetime index
    common_datetime = sorted(set().union(*[df.index for df in all_data]))

    # Reindex each DataFrame with the common datetime index
    reindexed_data = [df.reindex(common_datetime, fill_value=np.nan) for df in all_data]

    # Concatenate the DataFrames along the columns axis
    combined_df = pd.concat(reindexed_data, axis=1)

    initial_close_array = np.array(list(initial_close_prices.values()))
    
    pairwise_dfs = []
    for i in range(0, len(reindexed_data), 2):
        pair_df = pd.concat([reindexed_data[i], reindexed_data[i+1]], axis=1)
        pairwise_dfs.append(pair_df)
        
    # HISTORICAL RETURN DATA
    all_data_returns = []
    for file_path in file_paths:
        data = pd.read_parquet(base_path + file_path + '.parquet', engine='pyarrow')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.drop_duplicates(subset='datetime')

        data = data[(data['datetime'].dt.hour > 9) | ((data['datetime'].dt.hour == 9) & (data['datetime'].dt.minute >= 30))]
        data = data[data['datetime'].dt.hour < 16]

        data = data.set_index('datetime')[['close']]
        
        data['returns'] = data['close'].pct_change()
        data = data.drop(columns=['close'])
        
        data.dropna(inplace=True)
        
        data.columns = [file_path + '_Returns']
        all_data_returns.append(data)

    common_datetime_returns = sorted(set().union(*[df.index for df in all_data_returns]))
    reindexed_data_returns = [df.reindex(common_datetime_returns, fill_value=np.nan) for df in all_data_returns]
    combined_df_returns = pd.concat(reindexed_data_returns, axis=1)

    pairwise_dfs_returns = []

    for i in range(0, len(reindexed_data_returns), 2):
        pair_df = pd.concat([reindexed_data_returns[i], reindexed_data_returns[i + 1]], axis=1)
        pairwise_dfs_returns.append(pair_df)
        
    # PROCESSED HISTORICAL RETURN DATA
    processed_data_returns = []
    for file_path in file_paths:
        data = pd.read_parquet(base_path + file_path + '.parquet', engine='pyarrow')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.drop_duplicates(subset='datetime')

        data = data[(data['datetime'].dt.hour > 9) | ((data['datetime'].dt.hour == 9) & (data['datetime'].dt.minute >= 30))]
        data = data[data['datetime'].dt.hour < 16]

        data = data.set_index('datetime')[['close']]
        
        data['returns'] = data['close'].pct_change()
        data = data.drop(columns=['close'])
        
        # Remove the first row since it contains NaN
        data.dropna(inplace=True)
        
        # Clean the returns data
        data['returns'] = clean_data(data['returns'])
        
        # Filter out values that exceed 1 or are less than -1 (no more than %100 increase/decrease in minute)
        data = data[(data['returns'] <= 1) & (data['returns'] >= -1)]
        
        data.columns = [file_path + '_Returns']
        processed_data_returns.append(data)

    processed_combined_df_returns = pd.concat(processed_data_returns, axis=1, join='inner')
    
    columns = processed_combined_df_returns.columns
    processed_pairwise_dfs_returns = []

    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):
            pair_df = processed_combined_df_returns[[columns[i], columns[i + 1]]]
            processed_pairwise_dfs_returns.append(pair_df)
            
    # SCALED HISTORICAL RETURN DATA
    all_data_returns_scaled = []
    for file_path in file_paths:
        data = pd.read_parquet(base_path + file_path + '.parquet', engine='pyarrow')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.drop_duplicates(subset='datetime')

        data = data[(data['datetime'].dt.hour > 9) | ((data['datetime'].dt.hour == 9) & (data['datetime'].dt.minute >= 30))]
        data = data[data['datetime'].dt.hour < 16]

        data = data.set_index('datetime')[['close']]
        
        data['returns'] = data['close'].pct_change()
        data = data.drop(columns=['close'])
        
        data.dropna(inplace=True)
        
        # Add 1 to each value in the returns column
        data['returns'] = data['returns'] + 1
        
        data.columns = [file_path + '_Returns']
        all_data_returns_scaled.append(data)

    common_datetime_returns = sorted(set().union(*[df.index for df in all_data_returns_scaled]))
    reindexed_data_returns = [df.reindex(common_datetime_returns, fill_value=np.nan) for df in all_data_returns_scaled]
    combined_df_returns_scaled = pd.concat(reindexed_data_returns, axis=1)
    
    columns = combined_df_returns_scaled.columns  # Get the column names from the scaled DataFrame

    pairwise_dfs_returns_scaled = []
    for i in range(0, len(columns) - 1, 2):
        # Concatenating pairs of columns to form pairwise DataFrames
        if i + 1 < len(columns):  # Ensure there is a next column to pair with
            pair_df = combined_df_returns_scaled[[columns[i], columns[i + 1]]]
            pairwise_dfs_returns_scaled.append(pair_df)
            
    # NORMALIZED SCALED HISTORICAL RETURN DATA
    all_data_returns_scaled_normalized = []

    for file_path in file_paths:
        data = pd.read_parquet(base_path + file_path + '.parquet', engine='pyarrow')
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.drop_duplicates(subset='datetime')

        data = data[(data['datetime'].dt.hour > 9) | ((data['datetime'].dt.hour == 9) & (data['datetime'].dt.minute >= 30))]
        data = data[data['datetime'].dt.hour < 16]

        data = data.set_index('datetime')[['close']]
        
        data['returns'] = data['close'].pct_change()
        data.drop(columns=['close'], inplace=True)  # Drop 'close' after calculating returns
        
        data.dropna(inplace=True)
        
        data['returns'] = clean_data(data['returns'])
        
        data['returns'] = data['returns'] + 1
        
        # Filter out values that exceed 2 or are less than 0 (no more than %100 increase/decrease in minute)
        data = data[(data['returns'] <= 2) & (data['returns'] >= 0)]    
        
        data.columns = [file_path + '_Returns']
        all_data_returns_scaled_normalized.append(data)

    common_datetime_returns = sorted(set().union(*[df.index for df in all_data_returns_scaled_normalized]))
    reindexed_data_returns = [df.reindex(common_datetime_returns, fill_value=np.nan) for df in all_data_returns_scaled_normalized]
    combined_df_returns_scaled_normalized = pd.concat(reindexed_data_returns, axis=1)
    combined_df_returns_scaled_normalized = clean_data(combined_df_returns_scaled_normalized)

    normalization_values = {}

    # Normalize each column and store normalization parameters
    for column in combined_df_returns_scaled_normalized.columns:
        min_val = combined_df_returns_scaled_normalized[column].min()
        max_val = combined_df_returns_scaled_normalized[column].max()
        normalization_values[column] = {'min': min_val, 'max': max_val}
        
        # Normalize if valid range
        if max_val > min_val:
            combined_df_returns_scaled_normalized[column] = (combined_df_returns_scaled_normalized[column] - min_val) / (max_val - min_val)

    columns = combined_df_returns_scaled_normalized.columns  # Get the column names from the normalized DataFrame

    pairwise_dfs_returns_scaled_normalized = []
    for i in range(0, len(columns) - 1, 2):
        pair_df = combined_df_returns_scaled_normalized[[columns[i], columns[i + 1]]]
        pairwise_dfs_returns_scaled_normalized.append(pair_df)
        
    # TEMPORAL CONVOLUTIONAL NEURAL NETWORK
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size):
            
            super(Chomp1d, self).__init__()
            self.chomp_size = chomp_size

        def forward(self, x):
            
            return x[:, :, :-self.chomp_size].contiguous()
        
    class TemporalBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
            super(TemporalBlock, self).__init__()
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
            self.chomp1 = Chomp1d(padding)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()
            self.init_weights()

        def init_weights(self):
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

        def forward(self, x):
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return out, self.relu(out + res)
        
    # GENERATOR
    class Generator(nn.Module):
        def __init__(self, seq_len, input_dim=3, output_dim=2):
            
            super(Generator, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.tcn = nn.ModuleList([
                TemporalBlock(input_dim, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=2, padding=2),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=4, padding=4),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=8, padding=8),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=16, padding=16),
            ])
            self.last = nn.Conv1d(80, output_dim, kernel_size=1, stride=1, dilation=1)

        def forward(self, x):
            
            skip_layers = []
            for layer in self.tcn:
                skip, x = layer(x)
                skip_layers.append(skip)
            x = self.last(x + sum(skip_layers))
            return x

    # DISCRIMINATOR
    class Discriminator(nn.Module):
        def __init__(self, seq_len, conv_dropout=0.05, input_dim=2):
            
            super(Discriminator, self).__init__()
            self.seq_len = seq_len
            self.tcn = nn.ModuleList([
                TemporalBlock(input_dim, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=2, padding=2),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=4, padding=4),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=8, padding=8),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=16, padding=16),
            ])
            self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
            self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

        def forward(self, x):
            
            skip_layers = []
            for layer in self.tcn:
                skip, x = layer(x)
                skip_layers.append(skip)
            x = self.last(x + sum(skip_layers))
            x = x.view(x.size(0), -1)
            return self.to_prob(x).squeeze()
        
    # CENTRAL DISCRIMINATOR
    class CentralDiscriminator(nn.Module):
        def __init__(self, seq_len, input_dim=16):
            super(CentralDiscriminator, self).__init__()
            self.seq_len = seq_len
            self.tcn = nn.ModuleList([
                TemporalBlock(input_dim, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=1, padding=1),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=2, padding=2),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=4, padding=4),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=8, padding=8),
                TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=16, padding=16),
            ])
            self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)
            self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

        def forward(self, x):
            skip_layers = []
            for layer in self.tcn:
                skip, x = layer(x)
                skip_layers.append(skip)
            x = self.last(x + sum(skip_layers))
            x = x.view(x.size(0), -1)
            return self.to_prob(x).squeeze()
        
    # BIVARIATE W-TC-GAN TRAINING
    num_epochs_biv = 200
    nz_biv = 3
    # number of batches to be made from number of seuqneces generated
    batch_size_biv = 400 # (number of batches = # sequences / batch_size)
    clip_biv = 0.01
    lr_biv = 0.0001
    # length of sequences generated by dataloader
    seq_len_biv = 4000 # must be less than data used (# sequences = data_used - seq_len + 1)
    data_used_biv = 81328 # max is 81328

    # Create the directory if it doesn't exist
    save_dir_biv = 'SavedGenerators_1' # ========================================= CHANGE DIRECTORY ====================================
    os.makedirs(save_dir_biv, exist_ok=True)

    data_pairs = []
    for name1, name2 in file_pairs:
        df_combined = combined_df_returns_scaled_normalized[[f'{name1}_Returns', f'{name2}_Returns']].iloc[:data_used_biv]
        data_pairs.append((df_combined, name1, name2))
        
    for i, (df_combined, name1, name2) in enumerate(data_pairs):
        # Extract the normalized returns
        returns1 = df_combined[f'{name1}_Returns'].values
        returns2 = df_combined[f'{name2}_Returns'].values
        
        generator = Generator(seq_len_biv, input_dim=nz_biv, output_dim=2).to(device)
        discriminator = Discriminator(seq_len_biv, input_dim=2).to(device)
        
        # Apply weight initialization
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        
        disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr_biv)
        gen_optimizer = optim.RMSprop(generator.parameters(), lr=lr_biv)

        dataset = Loader32(returns1, returns2, seq_len_biv)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_biv)
        
        t = tqdm(range(num_epochs_biv))
        
        for epoch in t:
            print(f"Epoch number: {epoch} for pair {name1} and {name2}")
            
            for idx, data in enumerate(dataloader, 0):
                if len(data) == 0:
                    continue  # Skip empty batches
                
                print(f"Processing batch {idx + 1}/{len(dataloader)}")

                discriminator.zero_grad()
                real = data.to(device)
                batch_size_biv, _, seq_len_biv = real.size()
                noise = torch.randn(batch_size_biv, nz_biv, seq_len_biv, device=device)
                fake = generator(noise).detach()
                disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
                penalty = gradient_penalty(discriminator, real, fake)
                disc_loss += penalty

                if torch.isnan(disc_loss).any() or torch.isinf(disc_loss).any():
                    print("NaN or Inf in discriminator loss")
                    break

                disc_loss.backward()
                disc_optimizer.step()

                for dp in discriminator.parameters():
                    dp.data.clamp_(-clip_biv, clip_biv)
        
                if idx % 5 == 0:
                    generator.zero_grad()
                    gen_loss = -torch.mean(discriminator(generator(noise)))
                    
                    if torch.isnan(gen_loss).any() or torch.isinf(gen_loss).any():
                        print("NaN or Inf in generator loss")
                        break

                    gen_loss.backward()
                    gen_optimizer.step()
                    
                for name, param in generator.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in generator parameter: {name}")

                for name, param in discriminator.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient in discriminator parameter: {name}")
                    
            if 'disc_loss' in locals() and 'gen_loss' in locals():
                t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
                print(f"Epoch {epoch}: Discriminator Loss: {disc_loss.item()}, Generator Loss: {gen_loss.item()}")
            else:
                t.set_description('Epoch %d' % epoch)
                
        # Save each generator separately with unique names in the 'SavedGenerators' folder
        torch.save(generator.state_dict(), os.path.join(save_dir_biv, f'trained_generator_pair_{name1}_{name2}_epoch_{epoch}.pth'))
        print(f"Saved generator model for pair {name1} and {name2} at epoch {epoch} to {save_dir_biv}")
        
    # MULTIVARIATE W-TC-GAN TRAINING
    # these paramaters can be different from the bivariate case
    num_epochs_muv = 200
    batch_size_muv = 400 # number of sythetic paths to generate for training
    clip_muv = 0.01
    lr_muv = 0.0001

    # these parameters must be identical to the bivariate case (match the paramaters of the generators you are choosing to load)
    # seq_len_muv = seq_len_biv # length of sequences generated by dataloader
    seq_len_muv = 1000 # must be less than seq_len_biv since seq_len_biv becomes used_data_biv to use all available synthetic data
    biv_epoch = 4 # load set of generators with this epoch number
    data_used_muv = seq_len_biv # used all available synthetic data
    nz_muv = 3 # nz_biv # must match noise dimension used in SavedGenerators

    load_dir_muv = '/Users/oscar/Desktop/Quant study/MCD - Project 1/Scripts/v2/SavedGenerators_1' # ============ CHANGE DIRECTORY ====================================

    # Create the directory if it doesn't exist
    save_dir_muv = 'ModifiedSavedGenerators_1' # ========================================= CHANGE DIRECTORY ====================================
    os.makedirs(save_dir_muv, exist_ok=True)
    save_dir_muv_cd = 'SavedCentralDsicriminators_1' # ========================================= CHANGE DIRECTORY ====================================
    os.makedirs(save_dir_muv_cd, exist_ok=True)
