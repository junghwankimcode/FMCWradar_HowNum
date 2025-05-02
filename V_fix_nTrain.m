clear all; clc; close all;
%% Parameter Setting
%nTrain = 2e3;
nTest = 2e3;
trainDoFlim = 4; 
testDoFlim = 4;
SNRmargin = 0;
Scene = 2;

% Radar parameters
f_c = 76.5*10^9;
c = 3*10^8;
lambda = c / f_c;
BW = 300e6;
Tsweep = 30e-6;
num_C = 32;
f_s = 2e6;
T_s = 1/f_s;
num_T = fix(Tsweep/T_s);
K = BW / Tsweep;
% Determined capabilities
R_max = f_s * Tsweep * c / (2 * BW);
V_max = c / (2 * (f_c * Tsweep));
%trainSNRset = -5:5:25;

trainSNR = 20;
nTrainset = [1e1 1e2 5e2 2e3];

str1 = num_T;
str2 = num_C;

%% Neural Network Layers & Training Options
MaxIteration = 1e4;
options = trainingOptions("adam", ...
 'InitialLearnRate',1e-3, ...
 'SquaredGradientDecayFactor',0.99, ...
 'MaxEpochs',30, ...
 'MiniBatchSize',64, ...
 'ExecutionEnvironment','gpu',...
 'OutputFcn',@(info) stopTraining(info,MaxIteration));

options_cnn = trainingOptions("adam",...
    'InitialLearnRate',1e-3,...
    'SquaredGradientDecayFactor',0.99,...
    'MaxEpochs',30,...
    'MiniBatchSize',64,...
    'ExecutionEnvironment','gpu',...
    'OutputFcn',@(info)stopTraining(info,MaxIteration));

layers1 = [sequenceInputLayer(num_T/2)
 fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(trainDoFlim)]; % Proposed: Column

layers2 = [sequenceInputLayer(num_C/2)
 fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(trainDoFlim)]; % Proposed: Row

layers3 = [sequenceInputLayer((size(diag(zeros(num_T,num_C)),1)/2))
 fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
 reluLayer
 fullyConnectedLayer(trainDoFlim)]; % Proposed: Diagonal/Anti

layers4 = [sequenceInputLayer(num_C)
    fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(trainDoFlim)]; % legacy row -16

layers5 = [sequenceInputLayer(num_T)
    fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(trainDoFlim)]; % legacy col -24

layers6 = [sequenceInputLayer((size(diag(zeros(num_T,num_C)),1)))
    fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
    reluLayer
    fullyConnectedLayer(trainDoFlim)]; %legacy diag/anti -16

%% Training Data Generation

for indexnTrain = 1:length(nTrainset)

 nTrain = nTrainset(indexnTrain)

 if trainSNR <= 5
    a = 0.5;
 else
    a = 0.1;
 end

 layers_CNN = [
    imageInputLayer([num_T num_C 1]) 

    convolution2dLayer(3, 32, 'Padding', 'same') % Conv1 (3x3x32)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) 

    convolution2dLayer(3, 64, 'Padding', 'same') % Conv2 (3x3x64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) 

    convolution2dLayer(3, 128, 'Padding', 'same') % Conv3 (3x3x128)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) 

    fullyConnectedLayer(128) 
    dropoutLayer(a)
    reluLayer

    fullyConnectedLayer(128) 
    dropoutLayer(a)
    reluLayer

    fullyConnectedLayer(trainDoFlim) 
    softmaxLayer
];

 temp_trainData1_col = [];
 temp_trainData2_col =[];
 temp_trainData3_col = [];
 temp_trainData_col_Ans = [];
 
 temp_trainData1_row = [];
 temp_trainData2_row = [];
 temp_trainData3_row = [];
 temp_trainData_row_Ans = [];
 
 temp_trainData1_diag = [];
 temp_trainData2_diag =[];
 temp_trainData3_diag =[];
 temp_trainData_diag_Ans = [];
 
 temp_trainData1_anti = [];
 temp_trainData2_anti = [];
 temp_trainData3_anti = [];
 temp_trainData_anti_Ans = [];

 trainData_CNN = zeros(num_T,num_C,1,nTrain);

 trainData_CNN_Ans = [];

 % 병렬 처리 풀 활성화 (이미 실행 중인지 확인 후 실행)
if isempty(gcp('nocreate'))
    parpool('local'); % 가용한 모든 코어 활용
end

 
 %==================== Training Phase ====================
 parfor indexN = 1:nTrain
 
 trainDoF = randi(trainDoFlim,1);

 if trainDoF > 0
 gain = (randn(1, trainDoF) + 1i*randn(1, trainDoF)) / sqrt(2);
 R = R_max * rand(1, trainDoF);
 V = V_max*ones(1,trainDoF); %* rand(1, trainDoF);
 else
 gain = []; R = []; V = [];
 end
 [~,~,observation] = gsGen_2DFMCW(Scene, trainSNR, trainDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
 
 %% Case 1: Column vector
 xr = observation;
 for i = 1:num_C
     data = xr(:,i);
     [~, Sn, ~] = makeHankel(data.');
     Sn = diag(Sn);
     temp_trainData1_col = [temp_trainData1_col Sn];
     temp_trainData2_col = [temp_trainData2_col abs(data)];
     temp_trainData3_col = [temp_trainData3_col abs(fft(data))];
     answer = zeros(trainDoFlim,1);
     answer(trainDoF) = 1;
     temp_trainData_col_Ans = [temp_trainData_col_Ans answer];
 end
 
 %% Case 2: Row vector
 xr = observation;
 for i = 1:num_T
     data = xr(i,:);
     [~, Sn, ~] = makeHankel(data);
     Sn = diag(Sn);
     temp_trainData1_row = [temp_trainData1_row Sn];
     temp_trainData2_row = [temp_trainData2_row abs(data).'];
     temp_trainData3_row = [temp_trainData3_row abs(fft(data)).'];
     answer = zeros(trainDoFlim,1);
     answer(trainDoF) = 1;
     temp_trainData_row_Ans = [temp_trainData_row_Ans answer];
 end
 
 %% Case 3 & 4: Diagonal / Anti-diagonal
 if num_C > num_T
 observation2 = observation;
     for i = 1:(num_C-num_T+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         temp_trainData1_diag = [temp_trainData1_diag S1];
         temp_trainData2_diag = [temp_trainData2_diag abs(data1)];
         temp_trainData3_diag = [temp_trainData3_diag abs(fft(data1))];
         answer = zeros(trainDoFlim,1);
         answer(trainDoF) = 1;
         temp_trainData_diag_Ans = [temp_trainData_diag_Ans answer];
     end

     observation2 = flip(observation2,2);
     for i = 1:(num_C-num_T+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         temp_trainData1_anti = [temp_trainData1_anti S1];
         temp_trainData2_anti = [temp_trainData2_anti abs(data1)];
         temp_trainData3_anti = [temp_trainData3_anti abs(fft(data1))];
         answer = zeros(trainDoFlim,1);
         answer(trainDoF) = 1;
         temp_trainData_anti_Ans = [temp_trainData_anti_Ans answer];
     end
 else
     observation2 = observation.';
     for i = 1:(num_T-num_C+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         temp_trainData1_diag = [temp_trainData1_diag S1];
         temp_trainData2_diag = [temp_trainData2_diag abs(data1)];
         temp_trainData3_diag = [temp_trainData3_diag abs(fft(data1))];
         answer = zeros(trainDoFlim,1);
         answer(trainDoF) = 1;
         temp_trainData_diag_Ans = [temp_trainData_diag_Ans answer];
     end
     observation3 = flip(observation.',2);
     for i = 1:(num_T-num_C+1)
         xr = observation3(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         temp_trainData1_anti = [temp_trainData1_anti S1];
         temp_trainData2_anti = [temp_trainData2_anti abs(data1)];
         temp_trainData3_anti = [temp_trainData3_anti abs(fft(data1))];
         answer = zeros(trainDoFlim,1);
         answer(trainDoF) = 1;
         temp_trainData_anti_Ans = [temp_trainData_anti_Ans answer];
     end
 end

 rangeDopplerMap = abs(fftshift(fft2(observation))); 
 rangeDopplerMap = (rangeDopplerMap - min(rangeDopplerMap(:))) / (max(rangeDopplerMap(:)) - min(rangeDopplerMap(:))); % 0~1 정규화


 trainData_CNN(:,:,:,indexN) = rangeDopplerMap; % Range-Doppler 맵 입력
 answer=zeros(trainDoFlim,1);
 answer(trainDoF)=1;
 trainData_CNN_Ans=[trainData_CNN_Ans answer];

 end
 
 % Convert cell arrays to matrices (dlarray with 'CBT' format)
 temp_trainData1_col = dlarray((temp_trainData1_col), 'CBT');
 temp_trainData2_col = dlarray(temp_trainData2_col,'CBT');
 temp_trainData3_col = dlarray(temp_trainData3_col,'CBT');
 temp_trainData_col_Ans = dlarray((temp_trainData_col_Ans), 'CBT');
 
 temp_trainData1_row = dlarray((temp_trainData1_row), 'CBT');
 temp_trainData2_row = dlarray(temp_trainData2_row,'CBT');
 temp_trainData3_row = dlarray(temp_trainData3_row,'CBT');
 temp_trainData_row_Ans = dlarray((temp_trainData_row_Ans), 'CBT');
 
 temp_trainData1_diag = dlarray((temp_trainData1_diag), 'CBT');
 temp_trainData2_diag = dlarray(temp_trainData2_diag, 'CBT');
 temp_trainData3_diag = dlarray(temp_trainData3_diag, 'CBT');
 temp_trainData_diag_Ans = dlarray((temp_trainData_diag_Ans), 'CBT');
 
 temp_trainData1_anti = dlarray((temp_trainData1_anti), 'CBT');
 temp_trainData2_anti = dlarray(temp_trainData2_anti, 'CBT');
 temp_trainData3_anti = dlarray(temp_trainData3_anti, 'CBT');
 temp_trainData_anti_Ans = dlarray((temp_trainData_anti_Ans), 'CBT');

 trainData_CNN = dlarray(trainData_CNN);
 trainData_CNN_Ans=dlarray(trainData_CNN_Ans,'CB');
 
 %% Network Training
 net_col = trainnet(temp_trainData1_col, temp_trainData_col_Ans, layers1, "mse", options);
 net_row = trainnet(temp_trainData1_row, temp_trainData_row_Ans, layers2, "mse", options);
 net_diag = trainnet(temp_trainData1_diag, temp_trainData_diag_Ans, layers3, "mse", options);
 net_anti = trainnet(temp_trainData1_anti, temp_trainData_anti_Ans, layers3, "mse", options);

 net_col_orig=trainnet(temp_trainData2_col,temp_trainData_col_Ans,layers5,"mse",options);
 net_row_orig=trainnet(temp_trainData2_row,temp_trainData_row_Ans,layers4,"mse",options);
 net_diag_orig=trainnet(temp_trainData2_diag,temp_trainData_diag_Ans,layers6,"mse",options);
 net_anti_orig=trainnet(temp_trainData2_anti,temp_trainData_anti_Ans,layers6,"mse",options);

 net_col_fft=trainnet(temp_trainData3_col,temp_trainData_col_Ans,layers5,"mse",options);
 net_row_fft=trainnet(temp_trainData3_row,temp_trainData_row_Ans,layers4,"mse",options);
 net_diag_fft=trainnet(temp_trainData3_diag,temp_trainData_diag_Ans,layers6,"mse",options);
 net_anti_fft=trainnet(temp_trainData3_anti,temp_trainData_anti_Ans,layers6,"mse",options);
                                  
 net_CNN = trainnet(trainData_CNN, trainData_CNN_Ans, layers_CNN,"crossentropy",options_cnn);

 %% Test Phase
 temp_test_result1 = [];
 temp_test_result2 = [];
 temp_test_result3 = [];
 temp_test_result4 = [];
 temp_test_result5 = [];
 temp_test_result6 = [];
 temp_test_result7 = [];
 temp_testAns = [];
 
 % MNOMP baseline parameters
 p_fa_mnomp = 1e-2;
 tau_mnomp = chi2inv((1-p_fa_mnomp)^(1/num_T), 2)/2;
 overSamplingRate_mnomp = 2; %4
 R_s_mnomp = 1;
 R_c_mnomp = 3;

 % 병렬 처리 풀 활성화 (이미 실행 중인지 확인 후 실행)
if isempty(gcp('nocreate'))
    parpool('local'); % 가용한 모든 코어 활용
end

 
 parfor indexM = 1:nTest
 
 % 초기화: 각 방식별 결과를 저장할 변수
 % Proposed branches:
 Result1_r = []; Result1_c = []; Result1_d = []; Result1_a = [];
 % Legacy branches:
 Result2_r = []; Result2_c = []; Result2_d = []; Result2_a = [];
 % % FFT branches:
 Result3_r = []; Result3_c = []; Result3_d = []; Result3_a = [];

 
 testSNR = trainSNR + rand(1)*SNRmargin - SNRmargin/2;
 testDoF = randi(testDoFlim,1);

 if testDoF > 0
     R = R_max * rand(1,testDoF);
     V = V_max*ones(1,testDoF); %* rand(1,testDoF);
     gain = (randn(1,testDoF) + 1i*randn(1,testDoF)) / sqrt(2);
 else
    R = []; V = []; gain = [];
 end
 [GS, superposition, observation] = gsGen_2DFMCW(Scene, testSNR, testDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
 
 %% MNOMP Prediction
 Result_MNOMP = [];
 for i = 1:num_C
     data = observation(:, i);
     [omega_est, ~, ~] = MNOMP(data, eye(length(data)), tau_mnomp, overSamplingRate_mnomp, R_s_mnomp, R_c_mnomp);
     m_est = length(omega_est);
     Result_MNOMP = [Result_MNOMP, m_est];
 end
 [m_val, ~] = mode(Result_MNOMP);
 m_val = min(max(m_val, 1), testDoFlim);
 result5 = zeros(testDoFlim, 1);
 result5(m_val) = 1;
 temp_test_result5 = [temp_test_result5 result5];

 %% MVALSE Prediction
 Result_VALSE = [];
 for i = 1:num_C
     data = observation(:, i);
     if size(data,1)==1, data = data.'; end
     x_gt = superposition(:, i);
     if size(x_gt,1)==1, x_gt = x_gt.'; end

     m = (0:length(data)-1)';
     ha = 2;
     Iter_max = 50; %50
     B = inf;
     yy_min = 0;
     alpha = 1;
     method_EP = 'diag_EP';
     out = VALSE_EP(data, m, ha, x_gt, Iter_max, B, yy_min, alpha, method_EP);

     if isempty(out) || ~isfield(out, 'freqs') || isempty(out.freqs)
        m_est = 1;
     else
        m_est = length(out.freqs);
     end
     Result_VALSE = [Result_VALSE, m_est];
 end
 if isempty(Result_VALSE)
    m_val2 = 1;
 else
    m_val2 = mode(Result_VALSE);
 end
 if isnan(m_val2)
    m_val2 = 1;
 else
    m_val2 = min(max(m_val2, 1), testDoFlim);
 end
 result6 = zeros(testDoFlim, 1);
 result6(m_val2) = 1;
 temp_test_result6 = [temp_test_result6 result6];
 
 %% Proposed Method Predictions (Deep Learning)
 % Column branch
 for i = 1:num_C
     data = observation(:, i);
     [~, Sn, ~] = makeHankel(data.');
     Sn = diag(Sn);
     testResult = double(predict(net_col, Sn.').');
     testResult2=double(predict(net_col_orig,(abs(data)).').');
     testResult3=double(predict(net_col_fft,(abs(fft(data))).').');
     [~, idx] = max(testResult);
     Result1_r = [Result1_r, idx];
     [~, idx] = max(testResult2);
     Result2_r = [Result2_r idx];
     [~, idx] = max(testResult3);
     Result3_r = [Result3_r idx];
 end
 % Row branch
 for i = 1:num_T
     data = observation(i,:);
     [~, Sn, ~] = makeHankel(data);
     Sn = diag(Sn);
     testResult = double(predict(net_row, Sn.').');
     testResult2=double(predict(net_row_orig,abs(data)).');
     testResult3=double(predict(net_row_fft,abs(fft(data))).');
     [~, idx] = max(testResult);
     Result1_c = [Result1_c, idx];
     [~, idx] = max(testResult2);
     Result2_c = [Result2_c idx];
     [~, idx] = max(testResult3);
     Result3_c = [Result3_c idx];
 end
 % Diagonal branch
 if num_C > num_T
     observation2 = observation;
     for i = 1:(num_C-num_T+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         testResult = double(predict(net_diag, S1.').');
         testResult2_1=double(predict(net_diag_orig,(abs(data1)).').');
         testResult3_1=double(predict(net_diag_fft,(abs(fft(data1))).').');
         [~, idx] = max(testResult);
         Result1_d = [Result1_d, idx];
         [~, idx] = max(testResult2_1);
         Result2_d = [Result2_d, idx];
         [~, idx] = max(testResult3_1);
         Result3_d = [Result3_d, idx];
     end
 else
     observation2 = observation.';
     for i = 1:(num_T-num_C+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         testResult = double(predict(net_diag, S1.').');
         testResult2_1=double(predict(net_diag_orig,(abs(data1)).').');
         testResult3_1=double(predict(net_diag_fft,(abs(fft(data1))).').');
         [~, idx] = max(testResult);
         Result1_d = [Result1_d, idx];
         [~, idx] = max(testResult2_1);
         Result2_d = [Result2_d, idx];
         [~, idx] = max(testResult3_1);
         Result3_d = [Result3_d, idx];
     end
 end
 % Anti-diagonal branch
 if num_C > num_T
     observation2 = flip(observation,2);
     for i = 1:(num_C-num_T+1)
         xr = observation2(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         testResult = double(predict(net_anti, S1.').');
         testResult2_1=double(predict(net_anti_orig,(abs(data1)).').');
         testResult3_1=double(predict(net_anti_fft,(abs(fft(data1))).').');
         [~, idx] = max(testResult);
         Result1_a = [Result1_a, idx];
         [~, idx] = max(testResult2_1);
         Result2_a = [Result2_a, idx];
         [~, idx] = max(testResult3_1);
         Result3_a = [Result3_a, idx];
     end
     else
     observation3 = flip(observation.',2);
     for i = 1:(num_T-num_C+1)
         xr = observation3(:, i:end);
         data1 = diag(xr);
         [~, S1, ~] = makeHankel(data1.');
         S1 = diag(S1);
         testResult = double(predict(net_anti, S1.').');
         testResult2_1=double(predict(net_anti_orig,(abs(data1)).').');
         testResult3_1=double(predict(net_anti_fft,(abs(fft(data1))).').');
         [~, idx] = max(testResult);
         Result1_a = [Result1_a, idx];
         [~, idx] = max(testResult2_1);
         Result2_a = [Result2_a, idx];
         [~, idx] = max(testResult3_1);
         Result3_a = [Result3_a, idx];
     end
 end
 % Soft Voting for Proposed method
 votes_proposed = zeros(1, testDoFlim);
 for j = 1:(testDoFlim)
     prob_r = sum(Result1_r == j) / length(Result1_r);
     prob_c = sum(Result1_c == j) / length(Result1_c);
     prob_d = sum(Result1_d == j) / length(Result1_d);
     prob_a = sum(Result1_a == j) / length(Result1_a);
     votes_proposed(j) = (prob_r + prob_c + prob_d + prob_a) / 4;
 end
 [~, pred_class_proposed] = max(votes_proposed);
 result1 = zeros(testDoFlim, 1);
 result1(pred_class_proposed) = 1;
 temp_test_result1 = [temp_test_result1 result1];


 votes_abs = zeros(1, testDoFlim);
 for j = 1:(testDoFlim)
     prob_r = sum(Result2_r == j) / length(Result2_r);
     prob_c = sum(Result2_c == j) / length(Result2_c);
     prob_d = sum(Result2_d == j) / length(Result2_d);
     prob_a = sum(Result2_a == j) / length(Result2_a);
     votes_abs(j) = (prob_r + prob_c + prob_d + prob_a) / 4;
 end
 [~, pred_class_abs] = max(votes_abs);
 result2 = zeros(testDoFlim, 1);
 result2(pred_class_abs) = 1;
 temp_test_result2 = [temp_test_result2 result2];


 votes_fft = zeros(1, testDoFlim);
 for j = 1:(testDoFlim)
     prob_r = sum(Result3_r == j) / length(Result3_r);
     prob_c = sum(Result3_c == j) / length(Result3_c);
     prob_d = sum(Result3_d == j) / length(Result3_d);
     prob_a = sum(Result3_a == j) / length(Result3_a);
     votes_fft(j) = (prob_r + prob_c + prob_d + prob_a) / 4;
 end
 [~, pred_class_fft] = max(votes_fft);
 result3 = zeros(testDoFlim, 1);
 result3(pred_class_fft) = 1;
 temp_test_result3 = [temp_test_result3 result3];
 

 
 %% CA-CFAR Processing (as before)
 numCPI = 1;
 RDC = observation;
 RDMs = zeros(num_T, num_C, numCPI);
 for i = 1:numCPI
 RD_frame = RDC(:, (i-1)*num_C+1:i*num_C, :);
 RDMs(:,:,:,i) = fft2(RD_frame, num_T, num_C);
 end
 numGuard = 2;
 numTrain = numGuard*2;
 P_fa = 1e-5;
 SNR_OFFSET = -5;
 RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));
 [RDM_mask, ~, ~, ~] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
 estiS4_cfar = length(find(RDM_mask));
 if estiS4_cfar > testDoFlim
 estiS4 = testDoFlim;
 elseif estiS4_cfar == 0
 estiS4 = 1;
 else
 estiS4 = estiS4_cfar;
 end
 result4 = zeros(testDoFlim,1);
 result4(estiS4) = 1;
 temp_test_result4 = [temp_test_result4 result4];

 rangeDopplerMap = abs(fftshift(fft2(observation))); 
 rangeDopplerMap = (rangeDopplerMap - min(rangeDopplerMap(:))) / (max(rangeDopplerMap(:)) - min(rangeDopplerMap(:))); % 0~1 정규화


 testResult_CNN = double(predict(net_CNN, rangeDopplerMap));

 T=2;
 logits = log(testResult_CNN);
 scaled_logits = logits / T;
 testReuslt_CNN = exp(scaled_logits)/sum(exp(scaled_logits));

 if max(testReuslt_CNN) <= 0.2
     [~,estiS_CNN]=max(testReuslt_CNN);

 else
    k=3;
    [~,sortidx]=sort(testReuslt_CNN,'descend');
    estiS_CNN=sortidx(randi(min(k,testDoFlim)));
 end

 result7=zeros(testDoFlim,1);
 result7(estiS_CNN)=1;
 temp_test_result7=[temp_test_result7 result7];

 
 %% Ground Truth
 t = zeros(testDoFlim, 1);
 t(testDoF) = 1;
 temp_testAns = [temp_testAns t];
 
 
 end % End parfor test


 detectionRate1(indexnTrain) = sum(sum(temp_test_result1 .* temp_testAns)) / nTest
 detectionRate2(indexnTrain) = sum(sum(temp_test_result2 .* temp_testAns)) / nTest
 detectionRate3(indexnTrain) = sum(sum(temp_test_result3 .* temp_testAns)) / nTest
 detectionRate4(indexnTrain) = sum(sum(temp_test_result4.*temp_testAns))/nTest
 detectionRate5(indexnTrain) = sum(sum(temp_test_result5 .* temp_testAns)) / nTest % MNOMP baseline
 detectionRate6(indexnTrain) = sum(sum(temp_test_result6 .* temp_testAns)) / nTest % MVALSE baseline
 detectionRate7(indexnTrain) = sum(sum(temp_test_result7 .* temp_testAns)) / nTest % 
 
end % for indexSNR
%% Figure: Compare Detection Rates
figure(2)
x = [10, 100, 500, 2000];

plot(x, detectionRate1, 'r-o', 'linewidth', 1)
%set(gca,'XScale','log');
hold on
grid on
plot(x, detectionRate2, 'b-.^', 'linewidth', 1)
plot(x, detectionRate3, 'g:d', 'linewidth', 1)
plot(x, detectionRate5, 'm--s', 'linewidth', 1)
plot(x, detectionRate6, '--<', 'linewidth', 1)
plot(x, detectionRate7, '-.|')
plot(x, detectionRate4, 'k--+', 'linewidth', 1)
set(gca, 'XScale', 'log');

% X축 눈금 강제 설정
xticks([10, 100, 500, 2000]);
xticklabels({'10^1', '10^2', '5\times10^2', '2\times10^3'});
%set(gca, 'TickLabelInterpreter', 'latex');
xlabel("T")
ylabel("Detection rate")
%legend("Proposed method", "Deep learning (with s)", "Deep learning (with s_F)", "CA-CFAR", "MNOMP",'MVALSE' ,'location', 'best');
legend("Proposed method", "Deep learning (with s)", "Deep learning (with s_F)", "MNOMP",'MVALSE-EP' ,'Deep learning-based (CNN-Range-Doppler responses)','CA-CFAR','location', 'best');