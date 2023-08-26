%% Synthetic Data Generation by Very Basic 1-D GAN
% Developer: Seyed Muhammad Hossein Mousavi - (August 2023) - SUPSI
clear;
close all;
clc;
% Load the original dataset
load fisheriris.mat;
original_data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
real_data_mean = mean (original_data);
real_data_std = std (original_data);

% Define the generator and discriminator networks
generator = @(z) original_data; % Identity mapping for simplicity
discriminator = @(x) (x - original_data); % Z-score normalization

% Training parameters
num_samples = 500;
num_epochs = 4;
batch_size = 160;
learning_rate = 0.01;
% Each run generates samples equal with number of samples in the origianl
% data. So, 3 runs means original data * 3.
Runs= 5; 
for i=1:Runs
% Training loop
for epoch = 1:num_epochs
for batch = 1:num_samples/batch_size
% Generate noise samples for the generator
noise = randn(batch_size, 1);
% Generate synthetic data using the generator
synthetic_data = generator(noise);
% Train the discriminator to distinguish real from synthetic data
discriminator_loss = mean((discriminator(synthetic_data) - noise).^2);
% Update the generator to fool the discriminator
generator_loss = mean((discriminator(generator(noise)) - noise).^2);
% Update the generator and discriminator parameters
generator = @(z) generator(z) - learning_rate * generator_loss;
discriminator = @(x) discriminator(x) - learning_rate * discriminator_loss;
end
Run = [' Epoch "',num2str(epoch)];
disp(Run);
end
% Generate synthetic data using the trained generator
noise_samples = randn(num_samples, 1);
synthetic_data= generator(noise_samples);
Syn(i,:)=synthetic_data;
Run2 = [' Run "',num2str(Runs)];
disp(Run2);
end

%% Converting cell to matrix
S = size(Syn(Runs)); SO = size (meas); SF = SO (1,2); SO = SO (1,1); 
for i=1:Runs
Syn2{i}=reshape(Syn(i,:),[SO,SF]);
Syn2{i}(:,end+1)=Target; 
end
Synthetic3 = cell2mat(Syn2');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% Plot data and classes
Feature1=1;
Feature2=4;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,3,1)
area(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,3,2)
area(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,3,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,3,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,3,5)
histogram(meas, 'Normalization', 'probability', 'DisplayName', 'Original Data');
hold on;
histogram(SyntheticData, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
legend('Original','Synthetic')
subplot(3,3,6)
histogram(synthetic_data, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
hold on;
x_range = linspace(real_data_mean - 3 * real_data_std, real_data_mean + 3 * real_data_std, 100);
real_data_distribution = normpdf(x_range, real_data_mean, real_data_std);
plot(x_range, real_data_distribution, 'r', 'LineWidth', 2, 'DisplayName', 'Real Data Distribution');
legend();
xlabel('Value');
ylabel('Probability');
title('Real Data vs. Synthetic Data Distribution');
subplot(3,3,7)
boxchart(meas);title('Original');
subplot(3,3,8)
boxchart(SyntheticData);title('Synthetic');
subplot(3,3,9)
probplot(meas);title('Original');
hold on;
probplot(SyntheticData);title('Original and Synthetic');

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);
