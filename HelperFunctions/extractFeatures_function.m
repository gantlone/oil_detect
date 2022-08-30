function feature_table_all = extractFeatures_function(fds, window_length, window_overlap, resample_rate, reference_table, pca_rate)

warning off

num = 1;

overlap_length = window_length * window_overlap / 100;
step_length = window_length - overlap_length;

% extract features
feature_table_all = table();

while hasdata(fds)
     disp(num);
     PCG = read(fds);
     % check resample rate
     if resample_rate == 0
          PCG_new = PCG.data;
          fs = PCG.fs;
     else
          PCG_new = resample(PCG.data, resample_rate, PCG.fs);
          fs = resample_rate;
     end
     
     % filter the signal between 25 to 400 Hz
     %PCG_new = butterworth_low_pass_filter(PCG_new, 2, 400, fs, false);
     %PCG_new = butterworth_high_pass_filter(PCG_new, 2, 25, fs, false);
     
     if nnz(strcmp(reference_table.record_name, PCG.filename)) == 1
          % get the current label
          % current_class = string(reference_table(strcmp(reference_table.record_name, PCG.filename, 3), :).record_label);
          current_class = reference_table(strcmp(reference_table.record_name, PCG.filename), :).record_label;
          label = current_class;
          number_of_windows = floor( (length(PCG_new) - overlap_length*fs) / (fs * step_length));
          
          subsample_table = table();
          for iwin = 1:number_of_windows
               current_start_sample = (iwin - 1) * fs * step_length + 1;
               current_end_sample = current_start_sample + window_length * fs - 1;
               current_signal = PCG_new(current_start_sample:current_end_sample);
               
               % check max and min value of subsample
               min_flag = min(current_signal);
               max_flag = max(current_signal);
               if min_flag ~= max_flag
                    subsample_table = [subsample_table; array2table(current_signal')];
               end
          end
          
          raw_subsample = table2array(subsample_table);
          raw_subsample = raw_subsample';
          % PCA start
          [n,m] = size(raw_subsample);
          % With submean.
          subsample_submean = raw_subsample - ones(n,1) * mean(raw_subsample,1);
          % Without submean.
          %subsample_submean = subsample;
          
          [coeff,score,~] = pca(subsample_submean);
          %[U,PC,eigenVal] = princomp(subsample_submean);
          
          % Reconstruction of subsamples by pca.
          new_subsample = (coeff(:,1:floor(pca_rate*size(score,2)))*...
               score(:,1:floor(pca_rate*size(score,2)))')';
          
          [~,y]=size(new_subsample);
          
          new_sample = table();
          old_sample = table();
          
          for cnt = 1:m
               old_sample = [old_sample; array2table(raw_subsample(:,cnt))];
          end
          for cnt = 1:y
               new_sample = [new_sample; array2table(new_subsample(:,cnt))];
          end
          
          Difference = table2array(old_sample)-table2array(new_sample);
          
          feature_table = table();
          checkpoint = 0;
          for iwin2 = 1:length(Difference)/(window_length * fs)
               current_start_sample = (iwin2 - 1) * fs * window_length  + 1;
               current_end_sample = current_start_sample + window_length * fs - 1;
               current_signal = Difference(current_start_sample:current_end_sample);
               
               % To check Max. & Min. value of subsample
               min_flag=min(current_signal);
               max_flag=max(current_signal);
               if min_flag ~= max_flag
                    % Calculate mean value feature
                    feature_table.meanValue(iwin2-checkpoint, 1) = mean(current_signal);
                    
                    % Calculate median value feature
                    feature_table.medianValue(iwin2-checkpoint, 1) = median(current_signal);
                    
                    % Calculate standard deviation feature
                    feature_table.standardDeviation(iwin2-checkpoint, 1) = std(current_signal);
                    
                    % Calculate mean absolute deviation feature
                    feature_table.meanAbsoluteDeviation(iwin2-checkpoint, 1) = mad(current_signal);
                    
                    % Calculate signal 25th percentile feature
                    feature_table.quantile25(iwin2-checkpoint, 1) = quantile(current_signal, 0.25);
                    
                    % Calculate signal 75th percentile feature
                    feature_table.quantile75(iwin2-checkpoint, 1) = quantile(current_signal, 0.75);
                    
                    % Calculate signal inter quartile range feature
                    feature_table.signalIQR(iwin2-checkpoint, 1) = iqr(current_signal);
                    
                    % Calculate skewness of the signal values
                    feature_table.sampleSkewness(iwin2-checkpoint, 1) = skewness(current_signal);
                    
                    % Calculate kurtosis of the signal values
                    feature_table.sampleKurtosis(iwin2-checkpoint, 1) = kurtosis(current_signal);
                    
                    % Calculate Shannon's entropy value of the signal
                    feature_table.signalEntropy(iwin2-checkpoint, 1) = signal_entropy(current_signal');
                    
                    % Calculate spectral entropy of the signal
                    feature_table.spectralEntropy(iwin2-checkpoint, 1) = spectral_entropy(current_signal, fs, 256);
                    
                    % Extract features from the power spectrum
                    [maxfreq, maxval, maxratio] = dominant_frequency_features(current_signal, fs, 256, 0);
                    feature_table.dominantFrequencyValue(iwin2-checkpoint, 1) = maxfreq;
                    feature_table.dominantFrequencyMagnitude(iwin2-checkpoint, 1) = maxval;
                    feature_table.dominantFrequencyRatio(iwin2-checkpoint, 1) = maxratio;
                    
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
                    
                    [MFCCs, ~, ~] = mfcc(current_signal, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L);
                    feature_table.MFCC1(iwin2-checkpoint, 1) = MFCCs(1);
                    feature_table.MFCC2(iwin2-checkpoint, 1) = MFCCs(2);
                    feature_table.MFCC3(iwin2-checkpoint, 1) = MFCCs(3);
                    feature_table.MFCC4(iwin2-checkpoint, 1) = MFCCs(4);
                    feature_table.MFCC5(iwin2-checkpoint, 1) = MFCCs(5);
                    feature_table.MFCC6(iwin2-checkpoint, 1) = MFCCs(6);
                    feature_table.MFCC7(iwin2-checkpoint, 1) = MFCCs(7);
                    feature_table.MFCC8(iwin2-checkpoint, 1) = MFCCs(8);
                    feature_table.MFCC9(iwin2-checkpoint, 1) = MFCCs(9);
                    feature_table.MFCC10(iwin2-checkpoint, 1) = MFCCs(10);
                    feature_table.MFCC11(iwin2-checkpoint, 1) = MFCCs(11);
                    feature_table.MFCC12(iwin2-checkpoint, 1) = MFCCs(12);
                    feature_table.MFCC13(iwin2-checkpoint, 1) = MFCCs(13);
                    feature_table.class(iwin2-checkpoint, 1) = label;
                    feature_table.source{iwin2-checkpoint, 1} = PCG.filename;
               else
                    checkpoint = checkpoint + 1;
               end
          end
          
          feature_table_all = [feature_table_all; feature_table];
     end
     num = num + 1;
end
end
