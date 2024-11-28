clear all
clc

%% parameter setting part
%nTrain = 1e4;
nTrainVec=[1e1 1e2 1e3 1e4];
nTest = 1e4;

trainDoFlim = 4;
testDoFlim = 4;
SNRmargin = 0;

Scene=2;

%radar parameters
f_c = 76.5*10^9;
c = 3*10^8;
lambda = c / f_c;
BW = 30e6;
Tsweep = 10e-6;
num_C = 16;
f_s = 2.4e6;
T_s = 1/f_s; 
num_T = fix(Tsweep/T_s);
K = BW / Tsweep;

% determined capabilities
R_max = f_s * Tsweep * c / (2 * BW);
V_max = c / (2 * ( f_c * Tsweep));

%trainSNRset = [0:5:40];
trainSNR=20;


str1 = num_T;
str2 = num_C;

%% neural network layer & options setting part
MaxIteration=1e4;

options = trainingOptions("adam",...
     initialLearnRate=1e-3,...
     SquaredGradientDecayFactor=0.99,...
     MaxEpochs=30,...
     MiniBatchSize=64,...
     OutputFcn=@(info)stopTraining(info,MaxIteration));

layers1 = [sequenceInputLayer(num_T/2)
        fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(trainDoFlim)]; %col -12- proposed col

layers2 = [sequenceInputLayer(num_C/2)
        fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(str2*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(trainDoFlim)]; % proposed row - 8

layers3 = [sequenceInputLayer((size(diag(zeros(num_T,num_C)),1)/2))
        fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(str1*length(zeros(trainDoFlim,1)))
        reluLayer
        fullyConnectedLayer(trainDoFlim)]; % proposed diag/anti-8

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


%% number of signal received targers estimation part
for indexnTrain = 1:length(nTrainVec)

    nTrain=nTrainVec(indexnTrain)


    
    trainData1_col = []; trainData1_row = []; trainData1_diag = []; trainData1_anti = [];
    trainData_col_Ans = []; trainData_row_Ans = []; trainData_diag_Ans = []; trainData_anti_Ans = [];
    trainData2_col = []; trainData2_row = []; trainData2_diag = []; trainData2_anti = [];
    trainData3_col = []; trainData3_row = []; trainData3_diag = []; trainData3_anti = [];

% Training Phase
parfor indexN = 1:nTrain
    trainDoF = randi(trainDoFlim,1); 
    gain = (randn(1,trainDoF) + 1i*randn(1,trainDoF)) / sqrt(2);
    R=R_max*rand(1,trainDoF);
    V = V_max*rand(1,trainDoF);
    [~,~,observation] = gsGen_2DFMCW(Scene,trainSNR,trainDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
    
    % Case 1 : Column vector
    xr = observation;
    for i = 1:num_C
        data = xr(:,i);
        [~,Sn,~] = makeHankel(data.');
        Sn = diag(Sn);
        trainData1_col = [trainData1_col Sn];
        trainData2_col = [trainData2_col abs(data)];
        trainData3_col = [trainData3_col abs(fft(data))];
        answer = zeros(trainDoFlim,1);
        answer(trainDoF) = 1;
        trainData_col_Ans = [trainData_col_Ans answer];
    end

    % Case 2 : Row vector
    xr = observation;
    for i = 1:num_T
        data = xr(i,:);
        [~,Sn,~] = makeHankel(data);
        Sn = diag(Sn);
        trainData1_row = [trainData1_row Sn];
        trainData2_row = [trainData2_row abs(data).'];
        trainData3_row = [trainData3_row abs(fft(data)).'];
        answer = zeros(trainDoFlim,1);
        answer(trainDoF) = 1;
        trainData_row_Ans = [trainData_row_Ans answer];
    end

    
    if num_C > num_T  % Case 3 : diagonal vector %fat matrix
        observation2 = observation;
        for i = 1: (num_C-num_T+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1.');
            S1 = diag(S1);
            trainData1_diag = [trainData1_diag S1];
            trainData2_diag = [trainData2_diag abs(data1)];
            trainData3_diag = [trainData3_diag abs(fft(data1))];
            answer = zeros(trainDoFlim,1);
            answer(trainDoF) = 1;
            trainData_diag_Ans = [trainData_diag_Ans answer];
        end
        % Case 4 : Anti diagonal vector
        observation2 = flip(observation2,2);
        for i = 1:(num_C-num_T+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1.');
            S1 = diag(S1);
            trainData1_anti = [trainData1_anti S1];
            trainData2_anti = [trainData2_anti abs(data1)];
            trainData3_anti = [trainData3_anti abs(fft(data1))];
            answer = zeros(trainDoFlim,1);
            answer(trainDoF) = 1;
            trainData_anti_Ans = [trainData_anti_Ans answer];
        end
    else %num_T>num_C
        observation2 = transpose(observation);  % Case 3 : diagonal vector
        for i = 1: (num_T-num_C+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1.');
            S1 = diag(S1);
            trainData1_diag = [trainData1_diag S1];
            trainData2_diag = [trainData2_diag abs(data1)];
            trainData3_diag = [trainData3_diag abs(fft(data1))];
            answer = zeros(trainDoFlim,1);
            answer(trainDoF) = 1;
            trainData_diag_Ans = [trainData_diag_Ans answer];
        end
        % Case 4 : Anti diagonal vector
        observation3 = flip(transpose(observation),2);
        for i = 1: (num_T-num_C+1)
            xr = observation3(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1.');
            S1 = diag(S1);
            trainData1_anti = [trainData1_anti S1];
            trainData2_anti = [trainData2_anti abs(data1)];
            trainData3_anti = [trainData3_anti abs(fft(data1))];
            answer = zeros(trainDoFlim,1);
            answer(trainDoF) = 1;
            trainData_anti_Ans = [trainData_anti_Ans answer];
        end
    end

    
end
trainData1_col=dlarray(trainData1_col, 'CBT');
trainData2_col=dlarray(trainData2_col, 'CBT');
trainData3_col=dlarray(trainData3_col, 'CBT');
trainData_col_Ans=dlarray(trainData_col_Ans,'CBT');

trainData1_row=dlarray(trainData1_row, 'CBT');
trainData2_row=dlarray(trainData2_row, 'CBT');
trainData3_row=dlarray(trainData3_row, 'CBT');
trainData_row_Ans=dlarray(trainData_row_Ans,'CBT');

trainData1_diag=dlarray(trainData1_diag, 'CBT');
trainData2_diag=dlarray(trainData2_diag, 'CBT');
trainData3_diag=dlarray(trainData3_diag, 'CBT');
trainData_diag_Ans=dlarray(trainData_diag_Ans,'CBT');

trainData1_anti=dlarray(trainData1_anti, 'CBT');
trainData2_anti=dlarray(trainData2_anti, 'CBT');
trainData3_anti=dlarray(trainData3_anti, 'CBT');
trainData_anti_Ans=dlarray(trainData_anti_Ans,'CBT');

%% Proposed Method four neural network part
net_col=trainnet(trainData1_col,trainData_col_Ans,layers1,"mse",options);

net_row=trainnet(trainData1_row,trainData_row_Ans,layers2,"mse",options);

net_diag=trainnet(trainData1_diag,trainData_diag_Ans,layers3,"mse",options);

net_anti=trainnet(trainData1_anti,trainData_anti_Ans,layers3,"mse",options);

%% Original Channel model four neural network part
net_col_orig=trainnet(trainData2_col,trainData_col_Ans,layers5,"mse",options);

net_row_orig=trainnet(trainData2_row,trainData_row_Ans,layers4,"mse",options);

net_diag_orig=trainnet(trainData2_diag,trainData_diag_Ans,layers6,"mse",options);

net_anti_orig=trainnet(trainData2_anti,trainData_anti_Ans,layers6,"mse",options);

%% FFT proceed channel model four neural network part
net_col_fft=trainnet(trainData3_col,trainData_col_Ans,layers5,"mse",options);

net_row_fft=trainnet(trainData3_row,trainData_row_Ans,layers4,"mse",options);

net_diag_fft=trainnet(trainData3_diag,trainData_diag_Ans,layers6,"mse",options);

net_anti_fft=trainnet(trainData3_anti,trainData_anti_Ans,layers6,"mse",options);

%% Test Phase
testAns = [];

test_result1_mat = []; test_result2_mat = []; test_result3_mat = []; test_result4_mat_cfar=[];

parfor indexM = 1:nTest
    testSNR = trainSNR + rand(1)*SNRmargin - SNRmargin/2;
    testDoF = randi(testDoFlim,1);
    R=R_max*rand(1,testDoF);
    V = V_max*rand(1,testDoF);
    gain = (randn(1,testDoF) + 1i*randn(1,testDoF)) / sqrt(2);
    [GS,~,observation] = gsGen_2DFMCW(Scene,testSNR,testDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
    %proposed
    % 초기화 변수
    Result1_1_r = []; Result1_2_r = []; Result1_3_r = []; Result1_4_r = [];% all column vector use

    Result1_1_c = []; Result1_2_c = []; Result1_3_c = []; Result1_4_c = [];

    Result1_1_d = []; Result1_2_d = []; Result1_3_d = []; Result1_4_d = [];

    Result1_1_a = []; Result1_2_a = []; Result1_3_a = []; Result1_4_a = [];

    Result1_r = []; Result1_c = []; Result1_d = []; Result1_a = [];

    %legacy
    Result2_1_r = []; Result2_2_r = []; Result2_3_r = []; Result2_4_r = [];

    Result2_1_c = []; Result2_2_c = []; Result2_3_c = []; Result2_4_c = [];

    Result2_1_d = []; Result2_2_d = []; Result2_3_d = []; Result2_4_d = [];

    Result2_1_a = []; Result2_2_a = []; Result2_3_a = []; Result2_4_a = [];

    Result2_r = []; Result2_c = []; Result2_d = []; Result2_a = [];

    %fft
    Result3_1_r = []; Result3_2_r = []; Result3_3_r = []; Result3_4_r = [];
    
    Result3_1_c = []; Result3_2_c = []; Result3_3_c = []; Result3_4_c = [];
    
    Result3_1_d = []; Result3_2_d = []; Result3_3_d = []; Result3_4_d = [];
    
    Result3_1_a = []; Result3_2_a = []; Result3_3_a = []; Result3_4_a = [];

    Result3_r = []; Result3_c = []; Result3_d = []; Result3_a = [];

    % Case 1 : Column vector
    xr = observation;
    for i = 1:num_C
        data = xr(:,i);
        [~,Sn,~] = makeHankel(data.'); %data= column vector
        Sn = diag(Sn);
        testResult1=double(predict(net_col,Sn.').');
        testResult2=double(predict(net_col_orig,(abs(data)).').');
        testResult3=double(predict(net_col_fft,(abs(fft(data))).').');
        [ss,ii] = max(testResult1);
        Result1_r = [Result1_r ii]; % 최빈값 찾기 위한 index 저장
        [ss,ii] = max(testResult2);
        Result2_r = [Result2_r ii];
        [ss,ii] = max(testResult3);
        Result3_r = [Result3_r ii];
    end
    Result1_1_r=length(find(Result1_r == 1)); Result1_2_r=length(find(Result1_r == 2)); 
    Result1_3_r=length(find(Result1_r == 3)); Result1_4_r=length(find(Result1_r == 4));
    num1_probab_p_r=Result1_1_r/length(Result1_r);num2_probab_p_r=Result1_2_r/length(Result1_r);
    num3_probab_p_r=Result1_3_r/length(Result1_r); num4_probab_p_r=Result1_4_r/length(Result1_r);

    Result2_1_r=length(find(Result2_r == 1)); Result2_2_r=length(find(Result2_r == 2));
    Result2_3_r=length(find(Result2_r == 3)); Result2_4_r=length(find(Result2_r == 4));
    num1_probab_l_r=Result2_1_r/length(Result2_r); num2_probab_l_r=Result2_2_r/length(Result2_r);
    num3_probab_l_r=Result2_3_r/length(Result2_r); num4_probab_l_r=Result2_4_r/length(Result2_r);

    Result3_1_r=length(find(Result3_r == 1)); Result3_2_r=length(find(Result3_r == 2));
    Result3_3_r=length(find(Result3_r == 3)); Result3_4_r=length(find(Result3_r == 4));
    num1_probab_f_r=Result3_1_r/length(Result3_r); num2_probab_f_r=Result3_2_r/length(Result3_r);
    num3_probab_f_r=Result3_3_r/length(Result3_r); num4_probab_f_r=Result3_4_r/length(Result3_r);
    

    % Case 2 : Row vector
    xr = observation;
    for i = 1:num_T
        data = xr(i,:); %data= row vector
        [~,Sn,~] = makeHankel(data);
        Sn = diag(Sn);
        testResult1=double(predict(net_row,Sn.').');
        testResult2=double(predict(net_row_orig,abs(data)).');
        testResult3=double(predict(net_row_fft,abs(fft(data))).');
        [ss,ii] = max(testResult1);
        Result1_c = [Result1_c ii];
        [ss,ii] = max(testResult2);
        Result2_c = [Result2_c ii];
        [ss,ii] = max(testResult3);
        Result3_c = [Result3_c ii];
    end
    Result1_1_c=length(find(Result1_c == 1)); Result1_2_c=length(find(Result1_c == 2));
    Result1_3_c=length(find(Result1_c == 3)); Result1_4_c=length(find(Result1_c == 4));
    num1_probab_p_c=Result1_1_c/length(Result1_c); num2_probab_p_c=Result1_2_c/length(Result1_c);
    num3_probab_p_c=Result1_3_c/length(Result1_c); num4_probab_p_c=Result1_4_c/length(Result1_c);

    Result2_1_c=length(find(Result2_c == 1)); Result2_2_c=length(find(Result2_c == 2));
    Result2_3_c=length(find(Result2_c == 3)); Result2_4_c=length(find(Result2_c == 4));
    num1_probab_l_c=Result2_1_c/length(Result2_c); num2_probab_l_c=Result2_2_c/length(Result2_c);
    num3_probab_l_c=Result2_3_c/length(Result2_c); num4_probab_l_c=Result2_4_c/length(Result2_c);

    Result3_1_c=length(find(Result3_c == 1)); Result3_2_c=length(find(Result3_c == 2));
    Result3_3_c=length(find(Result3_c == 3)); Result3_4_c=length(find(Result3_c == 4));
    num1_probab_f_c=Result3_1_c/length(Result3_c); num2_probab_f_c=Result3_2_c/length(Result3_c);
    num3_probab_f_c=Result3_3_c/length(Result3_c); num4_probab_f_c=Result3_4_c/length(Result3_c);
    

    if num_C > num_T
        observation2 = observation; % Case 3 : Diagonal vector
        for i = 1:(num_C-num_T+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1); %data1=row vecor
            S1 = diag(S1);
            testResult1_1=double(predict(net_diag,S1.').');
            testResult2_1=double(predict(net_diag_orig,(abs(data1)).').');
            testResult3_1=double(predict(net_diag_fft,(abs(fft(data1))).').');
            [~,ii1] = max(testResult1_1);
            Result1_d = [Result1_d ii1];
            [~,ii1] = max(testResult2_1);
            Result2_d = [Result2_d ii1];
            [~,ii1] = max(testResult3_1);
            Result3_d = [Result3_d ii1];
        end
        Result1_1_d=length(find(Result1_d == 1)); Result1_2_d=length(find(Result1_d == 2));
        Result1_3_d=length(find(Result1_d == 3)); Result1_4_d=length(find(Result1_d == 4));
        num1_probab_p_d=Result1_1_d/length(Result1_d); num2_probab_p_d=Result1_2_d/length(Result1_d);
        num3_probab_p_d=Result1_3_d/length(Result1_d); num4_probab_p_d=Result1_4_d/length(Result1_d);

        Result2_1_d=length(find(Result2_d == 1));  Result2_2_d=length(find(Result2_d == 2));
        Result2_3_d=length(find(Result2_d == 3));  Result2_4_d=length(find(Result2_d == 4));
        num1_probab_l_d=Result2_1_d/length(Result2_d); num2_probab_l_d=Result2_2_d/length(Result2_d);
        num3_probab_l_d=Result2_3_d/length(Result2_d); num4_probab_l_d=Result2_4_d/length(Result2_d);

        Result3_1_d=length(find(Result3_d == 1)); Result3_2_d=length(find(Result3_d == 2));
        Result3_3_d=length(find(Result3_d == 3)); Result3_4_d=length(find(Result3_d == 4));
        num1_probab_f_d=Result3_1_d/length(Result3_d); num2_probab_f_d=Result3_2_d/length(Result3_d);
        num3_probab_f_d=Result3_3_d/length(Result3_d); num4_probab_f_d=Result3_4_d/length(Result3_d);
        
        % Case 4 : Anti diagonal vector
        observation2 = flip(observation2,2);
        for i = 1:(num_C-num_T+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1); % data1 = row vector
            S1 = diag(S1);
            testResult1_1=double(predict(net_anti,S1.').');
            testResult2_1=double(predict(net_anti_orig,(abs(data1)).').');
            testResult3_1=double(predict(net_anti_fft,(abs(fft(data1))).').');
            [~,ii1] = max(testResult1_1);
            Result1_a = [Result1_a ii1]; % Proposed
            [~,ii1] = max(testResult2_1);
            Result2_a = [Result2_a ii1]; % Original (Abs)
            [~,ii1] = max(testResult3_1);
            Result3_a = [Result3_a ii1]; % FFT proceed (Abs)->~: 각 열에서 최대 값, each 열에서 최대값 있는 index
        end
        Result1_1_a=length(find(Result1_a == 1));  Result1_2_a=length(find(Result1_a == 2));
        Result1_3_a=length(find(Result1_a == 3)); Result1_4_a=length(find(Result1_a == 4));
        num1_probab_p_a=Result1_1_a/length(Result1_a); num2_probab_p_a=Result1_2_a/length(Result1_a);
        num3_probab_p_a=Result1_3_a/length(Result1_a); num4_probab_p_a=Result1_4_a/length(Result1_a);

        Result2_1_a=length(find(Result2_a == 1));  Result2_2_a=length(find(Result2_a == 2));
        Result2_3_a=length(find(Result2_a == 3)); Result2_4_a=length(find(Result2_a == 4));
        num1_probab_l_a=Result2_1_a/length(Result2_a); num2_probab_l_a=Result2_2_a/length(Result2_a);
        num3_probab_l_a=Result2_3_a/length(Result2_a); num4_probab_l_a=Result2_4_a/length(Result2_a);

        Result3_1_a=length(find(Result3_a == 1)); Result3_2_a=length(find(Result3_a == 2));
        Result3_3_a=length(find(Result3_a == 3)); Result3_4_a=length(find(Result3_a == 4));
        num1_probab_f_a=Result3_1_a/length(Result3_a); num2_probab_f_a=Result3_2_a/length(Result3_a);
        num3_probab_f_a=Result3_3_a/length(Result3_a); num4_probab_f_a=Result3_4_a/length(Result3_a);
        
    else
        observation2 = transpose(observation); % Case 3 : Diagonal vector
        for i = 1:(num_T-num_C+1)
            xr = observation2(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1); %data1=row vecor
            S1 = diag(S1);
            testResult1_1=double(predict(net_diag,S1.').');
            testResult2_1=double(predict(net_diag_orig,(abs(data1)).').');
            testResult3_1=double(predict(net_diag_fft,(abs(fft(data1))).').');
            [~,ii1] = max(testResult1_1);
            Result1_d = [Result1_d ii1];
            [~,ii1] = max(testResult2_1);
            Result2_d = [Result2_d ii1];
            [~,ii1] = max(testResult3_1);
            Result3_d = [Result3_d ii1];
        end
        Result1_1_d=length(find(Result1_d == 1));  Result1_2_d=length(find(Result1_d == 2));
        Result1_3_d=length(find(Result1_d == 3));  Result1_4_d=length(find(Result1_d == 4));
        num1_probab_p_d=Result1_1_d/length(Result1_d);  num2_probab_p_d=Result1_2_d/length(Result1_d);
        num3_probab_p_d=Result1_3_d/length(Result1_d); num4_probab_p_d=Result1_4_d/length(Result1_d);

        Result2_1_d=length(find(Result2_d == 1)); Result2_2_d=length(find(Result2_d == 2));
        Result2_3_d=length(find(Result2_d == 3)); Result2_4_d=length(find(Result2_d == 4));
        num1_probab_l_d=Result2_1_d/length(Result2_d);  num2_probab_l_d=Result2_2_d/length(Result2_d);
        num3_probab_l_d=Result2_3_d/length(Result2_d);  num4_probab_l_d=Result2_4_d/length(Result2_d);

        Result3_1_d=length(find(Result3_d == 1));  Result3_2_d=length(find(Result3_d == 2));
        Result3_3_d=length(find(Result3_d == 3));  Result3_4_d=length(find(Result3_d == 4));
        num1_probab_f_d=Result3_1_d/length(Result3_d); num2_probab_f_d=Result3_2_d/length(Result3_d);
        num3_probab_f_d=Result3_3_d/length(Result3_d); num4_probab_f_d=Result3_4_d/length(Result3_d);
        
        % Case 4 : Anti diagonal vector
        observation3 = flip(transpose(observation),2);
        for i = 1:(num_T-num_C+1)
            xr = observation3(:,i:end);
            data1 = diag(xr);
            [~,S1,~] = makeHankel(data1); % data1 = row vector
            S1 = diag(S1);
            testResult1_1=double(predict(net_anti,S1.').');
            testResult2_1=double(predict(net_anti_orig,(abs(data1)).').');
            testResult3_1=double(predict(net_anti_fft,(abs(fft(data1))).').');
            [~,ii1] = max(testResult1_1);
            Result1_a = [Result1_a ii1]; % Proposed
            [~,ii1] = max(testResult2_1);
            Result2_a = [Result2_a ii1]; % Original (Abs)
            [~,ii1] = max(testResult3_1);
            Result3_a = [Result3_a ii1]; % FFT proceed (Abs)->~: 각 열에서 최대 값, each 열에서 최대값 있는 index
        end
        Result1_1_a=length(find(Result1_a == 1)); Result1_2_a=length(find(Result1_a == 2));
        Result1_3_a=length(find(Result1_a == 3)); Result1_4_a=length(find(Result1_a == 4));
        num1_probab_p_a=Result1_1_a/length(Result1_a); num2_probab_p_a=Result1_2_a/length(Result1_a);
        num3_probab_p_a=Result1_3_a/length(Result1_a); num4_probab_p_a=Result1_4_a/length(Result1_a);

        Result2_1_a=length(find(Result2_a == 1));  Result2_2_a=length(find(Result2_a == 2));
        Result2_3_a=length(find(Result2_a == 3));  Result2_4_a=length(find(Result2_a == 4));
        num1_probab_l_a=Result2_1_a/length(Result2_a); num2_probab_l_a=Result2_2_a/length(Result2_a);
        num3_probab_l_a=Result2_3_a/length(Result2_a); num4_probab_l_a=Result2_4_a/length(Result2_a);

        Result3_1_a=length(find(Result3_a == 1));  Result3_2_a=length(find(Result3_a == 2));
        Result3_3_a=length(find(Result3_a == 3));  Result3_4_a=length(find(Result3_a == 4));
        num1_probab_f_a=Result3_1_a/length(Result3_a); num2_probab_f_a=Result3_2_a/length(Result3_a);
        num3_probab_f_a=Result3_3_a/length(Result3_a); num4_probab_f_a=Result3_4_a/length(Result3_a);
    
    end

    %soft voting 
    %proposed
    num1_proposed_mean_probab = mean(num1_probab_p_r+num1_probab_p_c+num1_probab_p_d+num1_probab_p_a);
    num2_proposed_mean_probab = mean(num2_probab_p_r+num2_probab_p_c+num2_probab_p_d+num2_probab_p_a);
    num3_proposed_mean_probab = mean(num3_probab_p_r+num3_probab_p_c+num3_probab_p_d+num3_probab_p_a);
    num4_proposed_mean_probab = mean(num4_probab_p_r+num4_probab_p_c+num4_probab_p_d+num4_probab_p_a);

    proposed_prob_set = [num1_proposed_mean_probab num2_proposed_mean_probab num3_proposed_mean_probab num4_proposed_mean_probab];
    [~, estiS1] = max(proposed_prob_set);

    num1_legacy_mean_probab = mean(num1_probab_l_r+num1_probab_l_c+num1_probab_l_d+num1_probab_l_a);
    num2_legacy_mean_probab = mean(num2_probab_l_r+num2_probab_l_c+num2_probab_l_d+num2_probab_l_a);
    num3_legacy_mean_probab = mean(num3_probab_l_r+num3_probab_l_c+num3_probab_l_d+num3_probab_l_a);
    num4_legacy_mean_probab = mean(num4_probab_l_r+num4_probab_l_c+num4_probab_l_d+num4_probab_l_a);

    legacy_prob_set = [num1_legacy_mean_probab num2_legacy_mean_probab num3_legacy_mean_probab num4_legacy_mean_probab];
    [~, estiS2] = max(legacy_prob_set);

    num1_fft_mean_probab = mean(num1_probab_f_r+num1_probab_f_c+num1_probab_f_d+num1_probab_f_a);
    num2_fft_mean_probab = mean(num2_probab_f_r+num2_probab_f_c+num2_probab_f_d+num2_probab_f_a);
    num3_fft_mean_probab = mean(num3_probab_f_r+num3_probab_f_c+num3_probab_f_d+num3_probab_f_a);
    num4_fft_mean_probab = mean(num4_probab_f_r+num4_probab_f_c+num4_probab_f_d+num4_probab_f_a);

    fft_prob_set = [num1_fft_mean_probab num2_fft_mean_probab num3_fft_mean_probab num4_fft_mean_probab];
    [~, estiS3] = max(fft_prob_set);

    result1 = zeros(testDoFlim,1);
    result2 = zeros(testDoFlim,1);
    result3 = zeros(testDoFlim,1);
    result4 = zeros(testDoFlim,1); %ca-cfar

    result1(estiS1)=1;
    result2(estiS2)=1;
    result3(estiS3)=1;

    test_result1_mat = [test_result1_mat result1];
    test_result2_mat = [test_result2_mat result2];
    test_result3_mat = [test_result3_mat result3];
    t = zeros(testDoFlim,1);
    t(testDoF) = 1;
    testAns = [testAns t];

     %% Estimation of R & V Part
    % proposed

     % mean angles of real target signals
    angles1=cell(1,numel(GS));
    all_angles1=[];
    
    for i = numel(GS)
        angles1{i}=GS{i}.angle1;
        all_angles1 = [all_angles1 angles1{i}];
    end

    mean_angle1=mean(all_angles1);

    angles2=cell(1, numel(GS));
    all_angles2=[];

    for i = numel(GS)
        angles2{i}=GS{i}.angle2;
        all_angles2=[all_angles2 angles2{i}];
    end
    
    mean_angle2=mean(all_angles2);

    
    if estiS1 == testDoF

        R_ESTIMATION = zeros(1,testDoF);
        V_ESTIMATION = zeros(1,testDoF);
    
        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS1); 
    
        for k=1:estiS1
            R_ESTIMATION(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

    elseif estiS1 < testDoF
        R_ESTIMATION = zeros(1,testDoF);
        V_ESTIMATION = zeros(1,testDoF);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS1); 

        for k=1:estiS1
            R_ESTIMATION(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

        for j =1:(testDoF-estiS1)
            R_ESTIMATION(j+estiS1) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(mean_angle1)+2*pi,2*pi);
            V_ESTIMATION(j+estiS1) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(mean_angle2)+2*pi,2*pi);
        end

      

    else %estiS > testDoF
        R_ESTIMATION = zeros(1,estiS1);
        V_ESTIMATION = zeros(1,estiS1);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS1); 
    
        for k=1:testDoF
            R_ESTIMATION(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end
        %mean_r_esti=mean(R_ESTIMATION);
        %mean_v_esti=mean(V_ESTIMATION);

       
        for a=1:(estiS1-testDoF)
            R=[R 0];
            V=[V 0]; 
        %     R_ESTIMATION=[R_ESTIMATION mean_r_esti];
        %     V_ESTIMATION=[V_ESTIMATION mean_v_esti];
        end
        

    end
    
    
    R_performance(indexM,indexnTrain)=sqrt(mse(sort(R),sort(R_ESTIMATION)));
    V_performance(indexM,indexnTrain)=sqrt(mse(sort(V),sort(V_ESTIMATION)));

    % legacy
    

    if estiS2 == testDoF
        R_ESTIMATION_legacy = zeros(1,testDoF);
        V_ESTIMATION_legacy = zeros(1,testDoF);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS2);
    
        for k=1:estiS2
            R_ESTIMATION_legacy(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_legacy(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

        

    elseif estiS2 < testDoF
        R_ESTIMATION_legacy = zeros(1,testDoF);
        V_ESTIMATION_legacy = zeros(1,testDoF);
        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS2);
    
        for k=1:estiS2
            R_ESTIMATION_legacy(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_legacy(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

        for j =1:(testDoF-estiS2)
            R_ESTIMATION_legacy(j+estiS2) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(mean_angle1)+2*pi,2*pi);
            V_ESTIMATION_legacy(j+estiS2) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(mean_angle2)+2*pi,2*pi);
        end

    else
        R_ESTIMATION_legacy = zeros(1,estiS2);
        V_ESTIMATION_legacy = zeros(1,estiS2);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS2);
    
        for k=1:testDoF
            R_ESTIMATION_legacy(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_legacy(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

        % mean_r_esti_legacy=mean(R_ESTIMATION_legacy);
        % mean_v_esti_legacy=mean(V_ESTIMATION_legacy);
        % 
        for a=1:(estiS2-testDoF)
            R=[R 0];
            V=[V 0]; 
        %     R_ESTIMATION=[R_ESTIMATION mean_r_esti];
        %     V_ESTIMATION=[V_ESTIMATION mean_v_esti];
        end

    end
    R_performance_legacy(indexM,indexnTrain)=sqrt(mse(sort(R),sort(R_ESTIMATION_legacy)));
    V_performance_legacy(indexM,indexnTrain)=sqrt(mse(sort(V),sort(V_ESTIMATION_legacy)));
    

    %fft
    
    if estiS3 == testDoF
        R_ESTIMATION_fft = zeros(1,testDoF);
        V_ESTIMATION_fft = zeros(1,testDoF);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS3);
    
        for k=1:estiS3
            R_ESTIMATION_fft(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_fft(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end


    elseif estiS3 < testDoF
        R_ESTIMATION_fft = zeros(1,testDoF);
        V_ESTIMATION_fft = zeros(1,testDoF);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS3);
    
        for k=1:estiS3
            R_ESTIMATION_fft(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_fft(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end


        for j =1:(testDoF-estiS3)
            R_ESTIMATION_fft(j+estiS3) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(mean_angle1)+2*pi,2*pi);
            V_ESTIMATION_fft(j+estiS3) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(mean_angle2)+2*pi,2*pi);
        end

    
    else
        R_ESTIMATION_fft = zeros(1,estiS3);
        V_ESTIMATION_fft = zeros(1,estiS3);

        [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS3);
    
        for k=1:testDoF
            R_ESTIMATION_fft(k) = f_s*(c/(2*K))*(1/(2*pi))*mod(angle(ang{k}.angle1)+2*pi,2*pi);
            V_ESTIMATION_fft(k) = (lambda/(2*Tsweep))*(1/(2*pi))*mod(angle(ang{k}.angle2)+2*pi,2*pi);
        end

        % mean_r_esti_fft=mean(R_ESTIMATION_fft);
        % mean_v_esti_fft=mean(V_ESTIMATION_fft);
        % 
        for a=1:(estiS3-testDoF)
            R=[R 0];
            V=[V 0]; 
        %     R_ESTIMATION=[R_ESTIMATION mean_r_esti];
        %     V_ESTIMATION=[V_ESTIMATION mean_v_esti];
        end

    end

    R_performance_fft(indexM,indexnTrain)=sqrt(mse(sort(R),sort(R_ESTIMATION_fft)));
    V_performance_fft(indexM,indexnTrain)=sqrt(mse(sort(V),sort(V_ESTIMATION_fft)));

    
    %%ca_cfar R & V estimation part
    numCPI =1;

    RDC = observation; % radar data (complex values matrix)
    RDMs = zeros(num_T, num_C, numCPI); %range-doppler map

    for i = 1:numCPI
        RD_frame = RDC(:,(i-1)*num_C+1:i*num_C,:);
        RDMs(:,:,:,i) = fft2(RD_frame,num_T,num_C);
    end

    numGuard = 2; % # of guard cells
    numTrain = numGuard*2; % # of training cells
    P_fa = 1e-5; % desired false alarm rate 
    SNR_OFFSET = -5; % dB
    RDM_dB = 10*log10(abs(RDMs(:,:,1,1))/max(max(abs(RDMs(:,:,1,1)))));

    [RDM_mask, cfar_ranges, cfar_dopps, K2] = ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET);
    %RDM_mask에서 0이 아닌 값의 개수가 number of targets
    
    
    estiS4_cfar = length(find(RDM_mask));
    
    if estiS4_cfar > testDoFlim
        estiS4 = 1;
    elseif estiS4_cfar == 0
        estiS4 = 1;
    else
        estiS4 = estiS4_cfar;
    end

    esti_range_cfar=(R_max*(cfar_ranges/num_T)).';
    esti_vel_cfar=(V_max*(cfar_dopps/num_C)).';

    esti_range_cfar_f=zeros(1,testDoF);
    esti_vel_cfar_f=zeros(1,testDoF);

    %mean_range=mean(R);
    %mean_vel=mean(V);

    if isempty(esti_range_cfar)
        for l=1:testDoF
            esti_range_cfar_f(:,l)=(R_max); 
        end
        
    else
        if estiS4 == testDoF
            esti_range_cfar_f = esti_range_cfar; 
        else % estiS4 > testDoF
            esti_range_cfar_f = esti_range_cfar;
            
            for p=1:abs(estiS4-testDoF)
                esti_range_cfar_f=[esti_range_cfar_f (R_max)];
            end
        
        end
    end

    if isempty(esti_vel_cfar)
        for l=1:testDoF
            esti_vel_cfar_f(:,l)=(V_max);
        end
    else
        if estiS4 == testDoF
            esti_vel_cfar_f = esti_vel_cfar;
        else % estiS4 > testDoF
            esti_vel_cfar_f = esti_vel_cfar;
            for p=1:abs(estiS4-testDoF)
                esti_vel_cfar_f=[esti_vel_cfar_f (V_max)];
            end
        
        end
    end

    R_performance_cfar(indexM,indexnTrain)=sqrt(mse(sort(R),sort(esti_range_cfar_f)));
    V_performance_cfar(indexM,indexnTrain)=sqrt(mse(sort(V),sort(esti_vel_cfar_f)));

    result4(estiS4)=1;
    test_result4_mat_cfar = [test_result4_mat_cfar result4];
    

end

R_performance_final(indexnTrain)=mean(R_performance(:,indexnTrain))
R_performance_legacy_final(indexnTrain)=mean(R_performance_legacy(:,indexnTrain))
R_performance_fft_final(indexnTrain)=mean(R_performance_fft(:,indexnTrain))
R_performance_cfar_final(indexnTrain)=mean(R_performance_cfar(:,indexnTrain))

V_performance_final(indexnTrain)=mean(V_performance(:,indexnTrain))
V_performance_legacy_final(indexnTrain)=mean(V_performance_legacy(:,indexnTrain))
V_performance_fft_final(indexnTrain)=mean(V_performance_fft(:,indexnTrain))
V_performance_cfar_final(indexnTrain)=mean(V_performance_cfar(:,indexnTrain))


detectionRate1(indexnTrain) = sum(sum(test_result1_mat .* testAns)) / nTest
detectionRate2(indexnTrain) = sum(sum(test_result2_mat .* testAns)) / nTest
detectionRate3(indexnTrain) = sum(sum(test_result3_mat .* testAns)) / nTest
detectionRate4(indexnTrain) = sum(sum(test_result4_mat_cfar.*testAns))/nTest

end



%% figure part
figure(1)
semilogx(nTrainVec, R_performance_final,'ro-','LineWidth',1); hold on
semilogx(nTrainVec, R_performance_legacy_final,'b^-.','LineWidth',1); hold on
semilogx(nTrainVec, R_performance_fft_final,'gdiamond:','LineWidth',1); hold on
semilogx(nTrainVec,R_performance_cfar_final,'k--+','Linewidth',1); hold on
grid on
xticks([1e1 1e2 1e3 1e4])
xlabel('Number of training dataset')
ylabel('RMSE of range [m]')
legend("Proposed method","Deep learning (with s)","Deep learning (with s_F)","CA-CFAR",'location','best');
hold off

figure(2)
semilogx(nTrainVec, V_performance_final, 'ro-','LineWidth',1); hold on
semilogx(nTrainVec, V_performance_legacy_final,'b^-.','LineWidth',1); hold on
semilogx(nTrainVec, V_performance_fft_final,'gdiamond:','LineWidth',1); hold on
semilogx(nTrainVec,V_performance_cfar_final,'k--+','Linewidth',1); hold on
grid on
xticks([1e1 1e2 1e3 1e4])
xlabel('Number of training dataset')
ylabel('RMSE of velocity [m/s]')
legend("Proposed method","Deep learning (with s)","Deep learning (with s_F)","CA-CFAR",'location','best');
hold off


figure(3)
loglog(nTrainVec, R_performance_final./R_max, 'r-o','LineWidth',1); hold on
loglog(nTrainVec, R_performance_legacy_final./R_max,'b-.^','Linewidth',1); hold on
loglog(nTrainVec, R_performance_fft_final./R_max,'g:diamond','Linewidth',1); hold on
loglog(nTrainVec,R_performance_cfar_final./R_max,'k--+','Linewidth',1); hold on
grid on
xticks([1e1 1e2 1e3 1e4])
xlabel('Number of training dataset')
ylabel('Normalized range error (R_{rmse}/R_{max})')
legend("Proposed method","Deep learning (with s)","Deep learning (with s_F)","CA-CFAR",'location','best');
hold off

figure(4)
loglog(nTrainVec, V_performance_final./V_max, 'ro-','LineWidth',1); hold on
loglog(nTrainVec, V_performance_legacy_final./V_max,'b^-.','LineWidth',1); hold on
loglog(nTrainVec, V_performance_fft_final./V_max,'gdiamond:','LineWidth',1); hold on
loglog(nTrainVec,V_performance_cfar_final./V_max,'k--+','Linewidth',1); hold on
grid on
xticks([1e1 1e2 1e3 1e4])
xlabel('Number of training dataset')
ylabel('Normalized velocity error (V_{rmse}/{V_max})')
legend("Proposed method","Deep learning (with s)","Deep learning (with s_F)","CA-CFAR",'location','best');
hold off


figure(5)
semilogx(nTrainVec,detectionRate1,'r-o','linewidth',1)
hold on
grid on
semilogx(nTrainVec,detectionRate2,'b-.^','linewidth',1)
semilogx(nTrainVec,detectionRate3,'g:d','linewidth',1)
semilogx(nTrainVec,detectionRate4,'k--+','linewidth',1)
xticks([1e1 1e2 1e3 1e4])
xlabel("Number of training dataset")
ylabel("Detection rate")
legend("Proposed method","Deep learning (with s)","Deep learning (with s_F)","CA-CFAR",'location','best');
