clear all;
close all;
clc;

%% read data
normal_pop = xlsread('G:\oil_detect\popping.xlsx','normal_pop');
normal_pop = normal_pop(150677:end);   %%Removal of data below 1.5v

abnormal_pop_screw_oil = xlsread('G:\oil_detect\popping.xlsx','abnormal_pop_screw_oil');
abnormal_pop_screw_oil = abnormal_pop_screw_oil(10:end);
abnormal_pop_screw_water = xlsread('G:\oil_detect\popping.xlsx','abnormal_pop_screw_water');
abnormal_pop_screw_water = abnormal_pop_screw_water(515:end);

abnormal_pop_centre_water = xlsread('G:\oil_detect\popping.xlsx','abnormal_pop_centre_water');
abnormal_pop_centre_water = abnormal_pop_centre_water(2000:end);
% abnormal_pop_long_centre_water = xlsread('E:\oil_detect\popping.xlsx','abnormal_pop_long_centre_water');
% abnormal_pop_long_centre_water = abnormal_pop_long_centre_water(80000:end);   %%Removal of data below 1.5v
% abnormal_pop_long_centre_water2 = xlsread('E:\oil_detect\popping.xlsx','abnormal_pop_long_centre_water2');

%% preprocess
%預設參數
lowpass_f = 200;
fs = 977;
overlap_window = 2500;
window = 5000;

%異常螺絲訊號合併
abnormal_pop_screw_oil = lowpass(abnormal_pop_screw_oil,lowpass_f,fs);
abnormal_pop_screw_oil = abnormal_pop_screw_oil(10:end-10);
%abnormal_pop_screw_oil = combine_signal(abnormal_pop_screw_oil); %去除毛邊
framed_abnormal_screw_oil_signal = frame_signal(abnormal_pop_screw_oil, overlap_window, window);
framed_abnormal_screw_oil_label = cell(length(framed_abnormal_screw_oil_signal),1);
for o = 1:length(framed_abnormal_screw_oil_label)
    framed_abnormal_screw_oil_label{o,1} = 23;
end
% 
abnormal_pop_screw_water = lowpass(abnormal_pop_screw_water,lowpass_f,fs);
abnormal_pop_screw_water = abnormal_pop_screw_water(10:end-10);
% %abnormal_pop_screw_water = combine_signal(abnormal_pop_screw_water); %去除毛邊
framed_abnormal_screw_water_signal = frame_signal(abnormal_pop_screw_water, overlap_window, window);
framed_abnormal_screw_water_label = cell(length(framed_abnormal_screw_water_signal),1);
for o = 1:length(framed_abnormal_screw_water_label)
    framed_abnormal_screw_water_label{o,1} = 23;
end

%正常訊號+異常偏心訊號合併
 low_pass_normal_pop = lowpass(normal_pop,lowpass_f,fs);
 low_pass_normal_pop = low_pass_normal_pop(10:end-10);
%low_pass_normal_pop = combine_signal(low_pass_normal_pop(10:end-10)); %去除毛邊
 framed_normal_signal = frame_signal(low_pass_normal_pop, overlap_window, window); 
 framed_normal_label = cell(length(framed_normal_signal),1);
 for o = 1:length(framed_normal_signal)
     framed_normal_label{o,1} = 1;
 end

 abnormal_pop_centre_water = lowpass(abnormal_pop_centre_water,lowpass_f,fs);
 abnormal_pop_centre_water = abnormal_pop_centre_water(10:end-2000);
%abnormal_pop_centre_water = combine_signal(abnormal_pop_centre_water); %去除毛邊
 framed_abnormal_centre_signal = frame_signal(abnormal_pop_centre_water, overlap_window, window);
 framed_abnormal_centre_label = cell(length(framed_abnormal_centre_signal),1);
 for o = 1:length(framed_abnormal_centre_signal)
     framed_abnormal_centre_label{o,1} = 4;
 end
 
 signal = [framed_normal_signal;framed_abnormal_screw_oil_signal;framed_abnormal_screw_water_signal;framed_abnormal_centre_signal];
 label = [framed_normal_label;framed_abnormal_screw_oil_label;framed_abnormal_screw_water_label;framed_abnormal_centre_label];
% z=find(~isnan(abnormal_pop_long_centre_water));
% B=abnormal_pop_long_centre_water(z);
% abnormal_pop_long_centre_water = lowpass(B,lowpass_f,fs);
% abnormal_pop_long_centre_water = combine_signal(abnormal_pop_long_centre_water); %去除毛邊
% framed_abnormal_centre_2 = frame_signal(abnormal_pop_long_centre_water, overlap_window, window);
% abnormal_pop_long_centre_water2 = lowpass(abnormal_pop_long_centre_water2,lowpass_f,fs);
% abnormal_pop_long_centre_water2 = combine_signal(abnormal_pop_long_centre_water2); %去除毛邊
% framed_abnormal_centre_3 = frame_signal(abnormal_pop_long_centre_water2, overlap_window, window);
% framed_abnormal_centre_signal = [framed_abnormal_centre_1;framed_abnormal_centre_2;framed_abnormal_centre_3];
rand_num = randperm((length(framed_normal_label)+length(framed_abnormal_screw_oil_label)+length(framed_abnormal_screw_water_label)+length(framed_abnormal_centre_label)))';
rand_signal = signal(rand_num(1:end),:);
rand_label = cell2mat(label(rand_num(1:end),:));

%% feature
signal_feature = cell(size(signal));
feature_train = [];
for idx = 1:length(signal)
    disp("Feature extraction (training): #" + idx)
    train_signal = rand_signal{idx};
    % Calculate mean value feature
    meanValue = mean(train_signal);

    % Calculate median value feature
    medianValue = median(train_signal);

    % Calculate standard deviation feature
    standardDeviation = std(train_signal);

    % Calculate mean absolute deviation feature
    meanAbsoluteDeviation = mad(train_signal);

    % Calculate signal 25th percentile feature
    quantile25 = quantile(train_signal, 0.25);

    % Calculate signal 75th percentile feature
    quantile75= quantile(train_signal, 0.75);

    % Calculate signal inter quartile range feature
    signalIQR = iqr(train_signal);

    % Calculate skewness of the signal values
    sampleSkewness = skewness(train_signal);

    % Calculate kurtosis of the signal values
    sampleKurtosis = kurtosis(train_signal);

    % Calculate Shannon's entropy value of the signal
    signalEntropy = signal_entropy(train_signal');

    % Calculate spectral entropy of the signal
    spectralEntropy = spectral_entropy(train_signal, fs, 256);

    % Extract features from the power spectrum
    [maxfreq, maxval, maxratio] = dominant_frequency_features(train_signal, fs, 256, 0);
    dominantFrequencyValue = maxfreq;
    dominantFrequencyMagnitude = maxval;
    dominantFrequencyRatio = maxratio;

    % Extract wavelet features
    % REMOVED because didn't contribute much to final model (only 1 of
    % them was selected by NCA among the "important" features)

    % Extract Mel-frequency cepstral coefficients
    %Tw = window_length*1000;      % analysis frame duration (ms)
    Tw = 25;      % analysis frame duration (ms)
    Ts = 10;                % analysis frame shift (ms)
    alpha = 0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels
    C = 12;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 5;                 % lower frequency limit (Hz)
    HF = 500;               % upper frequency limit (Hz)

    [MFCCs, ~, ~] = mfcc(train_signal, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L);
    feature = [meanValue;medianValue;standardDeviation;meanAbsoluteDeviation;quantile25;
          quantile75;signalIQR;sampleSkewness;sampleKurtosis;signalEntropy;spectralEntropy;
           dominantFrequencyValue;dominantFrequencyMagnitude;dominantFrequencyRatio;MFCCs];
%       feature = [meanValue;medianValue;standardDeviation;meanAbsoluteDeviation;quantile25;
%            quantile75;signalIQR;sampleSkewness;sampleKurtosis;signalEntropy;MFCCs];
    %%normalize
    %feature = zscore(feature);
    feature_train = [feature_train,feature];
end
    %PCG_Features_train(idx,1) = {feature};
rand_feature = mapminmax(feature_train)';
% 
% %% predict
% train_rand_feature = rand_feature(floor(length(rand_feature)/5)+1:end,:);
% test_rand_feature = rand_feature(1:floor(length(rand_feature)/5),:);
% train_rand_label = rand_label(floor(length(rand_label)/5)+1:end,:);
% test_rand_label = rand_label(1:floor(length(rand_label)/5),:);

%% cv partition
prompt = 'choose your k value: ';
k = input(prompt);
c = cvpartition(rand_label,'KFold',k);
train_accuracy = 0;
test_accuracy = 0;
for k_number = 1:k
    train_index = training(c,k_number);
    test_index = test(c,k_number);
    train_feature = rand_feature(train_index,:);
    test_feature = rand_feature(test_index,:);
    train_label = rand_label(train_index,:);
    test_label = rand_label(test_index,:);

%% model
    t = templateSVM('KernelFunction','gaussian');
    Md1 = fitcecoc(train_feature,train_label,'Learners',t);
%     Md1 = fitcecoc(train_feature,train_label,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',6),'Learners','svm');
%     opts = struct('CVPartition',c,'AcquisitionFunctionName','expected-improvement-plus');
%     SVMModel = fitcsvm(rand_feature,rand_label,'KernelFunction','gaussian',...
%         'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
    %train 
    train_predict_label = predict(Md1, train_feature);
    train_acc = sum(train_predict_label == train_label)/length(train_label)*100;
    train_accuracy = train_accuracy + train_acc;
    figure
    confusionchart(train_label,train_predict_label,'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Training')
    
    %test
    test_predict_label = predict(Md1, test_feature);
    test_acc = sum(test_predict_label == test_label)/length(test_label)*100;
    test_accuracy = test_accuracy + test_acc;
    figure
    confusionchart(test_label,test_predict_label,'RowSummary','row-normalized','ColumnSummary','column-normalized','Title','Testing')
    
end

train_ave_acc = train_accuracy/k;
test_ave_acc = test_accuracy/k;



