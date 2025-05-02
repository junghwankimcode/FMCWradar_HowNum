clear all
clc
%% parameter setting part
nTrain = 2e3;
nTest = 2e3;
%str = 12;
%trainDoFlim = 4;
%testDoFlim =4;
trainDoFlimset = [4, 6, 8, 10, 12];
SNRmargin = 0;
Scene=2;
%radar parameters
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
% determined capabilities
R_max = f_s * Tsweep * c / (2 * BW);
V_max = c / (2 * ( f_c * Tsweep));
%trainSNRset = [20];
trainSNR = 20;
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
for indexDoFlim = 1:length(trainDoFlimset)
 trainDoFlim = trainDoFlimset(indexDoFlim)
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
 
 trainData1_col = [];
 trainData1_row = [];
 trainData1_diag = [];
 trainData1_anti = [];
 trainData_col_Ans = [];
 trainData_row_Ans = [];
 trainData_diag_Ans = [];
 trainData_anti_Ans = [];
 
% Training Phase
parfor indexN = 1:nTrain
 trainDoF = randi(trainDoFlim,1); 
 gain = (randn(1,trainDoF) + 1i*randn(1,trainDoF)) / sqrt(2);
 R=R_max*rand(1,trainDoFlim);
 V = V_max*rand(1,trainDoFlim);
 [~,~,observation] = gsGen_2DFMCW(Scene,trainSNR,trainDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
 
 % Case 1 : Column vector
 xr = observation;
 for i = 1:num_C
 data = xr(:,i);
 [~,Sn,~] = makeHankel(data.');
 Sn = diag(Sn);
 trainData1_col = [trainData1_col Sn];
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
 answer = zeros(trainDoFlim,1);
 answer(trainDoF) = 1;
 trainData_row_Ans = [trainData_row_Ans answer];
 end
 
 if num_C > num_T % Case 3 : diagonal vector %fat matrix
 observation2 = observation;
 for i = 1: (num_C-num_T+1)
 xr = observation2(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1.');
 S1 = diag(S1);
 trainData1_diag = [trainData1_diag S1];
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
 answer = zeros(trainDoFlim,1);
 answer(trainDoF) = 1;
 trainData_anti_Ans = [trainData_anti_Ans answer];
 end
 else %num_T>num_C
 observation2 = transpose(observation); % Case 3 : diagonal vector
 for i = 1: (num_T-num_C+1)
 xr = observation2(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1.');
 S1 = diag(S1);
 trainData1_diag = [trainData1_diag S1];
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
 answer = zeros(trainDoFlim,1);
 answer(trainDoF) = 1;
 trainData_anti_Ans = [trainData_anti_Ans answer];
 end
 end
 
end
trainData1_col=dlarray(trainData1_col, 'CBT');
trainData_col_Ans=dlarray(trainData_col_Ans,'CBT');
trainData1_row=dlarray(trainData1_row, 'CBT');
trainData_row_Ans=dlarray(trainData_row_Ans,'CBT');
trainData1_diag=dlarray(trainData1_diag, 'CBT');
trainData_diag_Ans=dlarray(trainData_diag_Ans,'CBT');
trainData1_anti=dlarray(trainData1_anti, 'CBT');
trainData_anti_Ans=dlarray(trainData_anti_Ans,'CBT');
%% Proposed Method four neural network part
net_col=trainnet(trainData1_col,trainData_col_Ans,layers1,"mse",options);
net_row=trainnet(trainData1_row,trainData_row_Ans,layers2,"mse",options);
net_diag=trainnet(trainData1_diag,trainData_diag_Ans,layers3,"mse",options);
net_anti=trainnet(trainData1_anti,trainData_anti_Ans,layers3,"mse",options);
%% Test Phase
testAns = [];
test_result1_mat = [];
test_result2_mat = [];
test_result3_mat = [];
test_result4_mat = [];
test_result5_mat = [];
testDoFlim = trainDoFlim;
parfor indexM = 1:nTest
 testSNR = trainSNR + rand(1)*SNRmargin - SNRmargin/2;
 testDoF = randi(testDoFlim,1);
 R=R_max*rand(1,testDoF);
 V = V_max*rand(1,testDoF);
 gain = (randn(1,testDoF) + 1i*randn(1,testDoF)) / sqrt(2);
 [GS,~,observation] = gsGen_2DFMCW(Scene,testSNR,testDoF, num_T, num_C, gain, 2*R, V, c, f_c, f_s, Tsweep, K);
 %proposed
 % 초기화 변수
 Result1_1_r = []; % all column vector use
 Result1_2_r = [];
 Result1_3_r = [];
 Result1_4_r = [];
 Result1_1_c = [];
 Result1_2_c = [];
 Result1_3_c = [];
 Result1_4_c = [];
 Result1_1_d = [];
 Result1_2_d = [];
 Result1_3_d = [];
 Result1_4_d = [];
 Result1_1_a = [];
 Result1_2_a = [];
 Result1_3_a = [];
 Result1_4_a = [];
 Result1_r = [];
 Result1_c = [];
 Result1_d = [];
 Result1_a = [];
 proposed_prob_set_r=[];
 proposed_prob_set_c=[];
 proposed_prob_set_d=[];
 proposed_prob_set_a=[];
 
 % Case 1 : Column vector
 xr = observation;
 for i = 1:num_C
 data = xr(:,i);
 [~,Sn,~] = makeHankel(data.'); %data= column vector
 Sn = diag(Sn);
 testResult1=double(predict(net_col,Sn.').');
 [ss,ii] = max(testResult1);
 Result1_r = [Result1_r ii]; % 최빈값 찾기 위한 index 저장
 end
 Result1_1_r=length(find(Result1_r == 1)); 
 Result1_2_r=length(find(Result1_r == 2));
 Result1_3_r=length(find(Result1_r == 3));
 Result1_4_r=length(find(Result1_r == 4));
 num1_probab_p_r=Result1_1_r/length(Result1_r);
 num2_probab_p_r=Result1_2_r/length(Result1_r);
 num3_probab_p_r=Result1_3_r/length(Result1_r);
 num4_probab_p_r=Result1_4_r/length(Result1_r);
 % Case 2 : Row vector
 xr = observation;
 for i = 1:num_T
 data = xr(i,:); %data= row vector
 [~,Sn,~] = makeHankel(data);
 Sn = diag(Sn);
 testResult1=double(predict(net_row,Sn.').');
 [ss,ii] = max(testResult1);
 Result1_c = [Result1_c ii];
 end
 Result1_1_c=length(find(Result1_c == 1)); 
 Result1_2_c=length(find(Result1_c == 2));
 Result1_3_c=length(find(Result1_c == 3));
 Result1_4_c=length(find(Result1_c == 4));
 num1_probab_p_c=Result1_1_c/length(Result1_c);
 num2_probab_p_c=Result1_2_c/length(Result1_c);
 num3_probab_p_c=Result1_3_c/length(Result1_c);
 num4_probab_p_c=Result1_4_c/length(Result1_c);
 if num_C > num_T
 observation2 = observation; % Case 3 : Diagonal vector
 for i = 1:(num_C-num_T+1)
 xr = observation2(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1); %data1=row vecor
 S1 = diag(S1);
 testResult1_1=double(predict(net_diag,S1.').');
 [~,ii1] = max(testResult1_1);
 Result1_d = [Result1_d ii1];
 
 end
 Result1_1_d=length(find(Result1_d == 1)); 
 Result1_2_d=length(find(Result1_d == 2));
 Result1_3_d=length(find(Result1_d == 3));
 Result1_4_d=length(find(Result1_d == 4));
 num1_probab_p_d=Result1_1_d/length(Result1_d);
 num2_probab_p_d=Result1_2_d/length(Result1_d);
 num3_probab_p_d=Result1_3_d/length(Result1_d);
 num4_probab_p_d=Result1_4_d/length(Result1_d);
 
 % Case 4 : Anti diagonal vector
 observation2 = flip(observation2,2);
 for i = 1:(num_C-num_T+1)
 xr = observation2(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1); % data1 = row vector
 S1 = diag(S1);
 testResult1_1=double(predict(net_anti,S1.').');
 [~,ii1] = max(testResult1_1);
 Result1_a = [Result1_a ii1]; % Proposed
 
 end
 Result1_1_a=length(find(Result1_a == 1)); 
 Result1_2_a=length(find(Result1_a == 2));
 Result1_3_a=length(find(Result1_a == 3));
 Result1_4_a=length(find(Result1_a == 4));
 num1_probab_p_a=Result1_1_a/length(Result1_a);
 num2_probab_p_a=Result1_2_a/length(Result1_a);
 num3_probab_p_a=Result1_3_a/length(Result1_a);
 num4_probab_p_a=Result1_4_a/length(Result1_a);
 
 
 else
 observation2 = transpose(observation); % Case 3 : Diagonal vector
 for i = 1:(num_T-num_C+1)
 xr = observation2(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1); %data1=row vecor
 S1 = diag(S1);
 testResult1_1=double(predict(net_diag,S1.').');
 [~,ii1] = max(testResult1_1);
 Result1_d = [Result1_d ii1];
 
 end
 Result1_1_d=length(find(Result1_d == 1)); 
 Result1_2_d=length(find(Result1_d == 2));
 Result1_3_d=length(find(Result1_d == 3));
 Result1_4_d=length(find(Result1_d == 4));
 num1_probab_p_d=Result1_1_d/length(Result1_d);
 num2_probab_p_d=Result1_2_d/length(Result1_d);
 num3_probab_p_d=Result1_3_d/length(Result1_d);
 num4_probab_p_d=Result1_4_d/length(Result1_d);
 
 % Case 4 : Anti diagonal vector
 observation3 = flip(transpose(observation),2);
 for i = 1:(num_T-num_C+1)
 xr = observation3(:,i:end);
 data1 = diag(xr);
 [~,S1,~] = makeHankel(data1); % data1 = row vector
 S1 = diag(S1);
 testResult1_1=double(predict(net_anti,S1.').');
 [~,ii1] = max(testResult1_1);
 Result1_a = [Result1_a ii1]; % Proposed
 
 end
 Result1_1_a=length(find(Result1_a == 1)); 
 Result1_2_a=length(find(Result1_a == 2));
 Result1_3_a=length(find(Result1_a == 3));
 Result1_4_a=length(find(Result1_a == 4));
 num1_probab_p_a=Result1_1_a/length(Result1_a);
 num2_probab_p_a=Result1_2_a/length(Result1_a);
 num3_probab_p_a=Result1_3_a/length(Result1_a);
 num4_probab_p_a=Result1_4_a/length(Result1_a);
 
 end
 %soft voting 
 %proposed
 % num1_proposed_mean_probab = mean(num1_probab_p_r+num1_probab_p_c+num1_probab_p_d+num1_probab_p_a);
 % num2_proposed_mean_probab = mean(num2_probab_p_r+num2_probab_p_c+num2_probab_p_d+num2_probab_p_a);
 % num3_proposed_mean_probab = mean(num3_probab_p_r+num3_probab_p_c+num3_probab_p_d+num3_probab_p_a);
 % num4_proposed_mean_probab = mean(num4_probab_p_r+num4_probab_p_c+num4_probab_p_d+num4_probab_p_a);
 % proposed_prob_set = [num1_proposed_mean_probab num2_proposed_mean_probab num3_proposed_mean_probab num4_proposed_mean_probab];
 % [~, estiS5] = max(proposed_prob_set);


 % proposed_prob_set_r = [num1_probab_p_r num2_probab_p_r num3_probab_p_r num4_probab_p_r];
 % [~, estiS1] = max(proposed_prob_set_r);
 % proposed_prob_set_c = [num1_probab_p_c num2_probab_p_c num3_probab_p_c num4_probab_p_c];
 % [~, estiS2] = max(proposed_prob_set_c);
 % proposed_prob_set_d = [num1_probab_p_d num2_probab_p_d num3_probab_p_d num4_probab_p_d];
 % [~, estiS3] = max(proposed_prob_set_d);
 % proposed_prob_set_a = [num1_probab_p_a num2_probab_p_a num3_probab_p_a num4_probab_p_a];
 % [~, estiS4] = max(proposed_prob_set_a);

 % Soft Voting for Proposed method
 votes_proposed = zeros(1, testDoFlim);
 for j = 1:(testDoFlim)
     prob_r = sum(Result1_r == j) / length(Result1_r);
     prob_c = sum(Result1_c == j) / length(Result1_c);
     prob_d = sum(Result1_d == j) / length(Result1_d);
     prob_a = sum(Result1_a == j) / length(Result1_a);
     votes_proposed(j) = (prob_r + prob_c + prob_d + prob_a) / 4;
 end
 [~, estiS5] = max(votes_proposed);
 result5 = zeros(testDoFlim, 1);
 result5(estiS5) = 1;

 vor= zeros(1,testDoFlim);
 for j = 1:(testDoFlim)
     prob_r = sum(Result1_r == j) / length(Result1_r);
     
     vor(j) = prob_r;
 end

 [~, estiS1] = max(vor);
 result1 = zeros(testDoFlim, 1);
 result1(estiS1) = 1;


 voc= zeros(1,testDoFlim);
 for j = 1:(testDoFlim)
     prob_c = sum(Result1_c == j) / length(Result1_c);
     
     voc(j) = prob_c;
 end

 [~, estiS2] = max(voc);
 result2 = zeros(testDoFlim, 1);
 result2(estiS2) = 1;

 vod= zeros(1,testDoFlim);
 for j = 1:(testDoFlim)
     prob_d = sum(Result1_d == j) / length(Result1_d);
     
     vod(j) = prob_d;
 end

 [~, estiS3] = max(vod);
 result3 = zeros(testDoFlim, 1);
 result3(estiS3) = 1;


 voa= zeros(1,testDoFlim);
 for j = 1:(testDoFlim)
     prob_a = sum(Result1_a == j) / length(Result1_a);
     
     voa(j) = prob_a;
 end

 [~, estiS4] = max(voa);
 result4 = zeros(testDoFlim, 1);
 result4(estiS4) = 1;
 
 % result1 = zeros(testDoFlim,1);
 % result2 = zeros(testDoFlim,1);
 % result3 = zeros(testDoFlim,1);
 % result4 = zeros(testDoFlim,1);
 % 
 % result1(estiS1)=1;
 % result2(estiS2)=1;
 % result3(estiS3)=1;
 % result4(estiS4)=1;

 test_result1_mat = [test_result1_mat result1];
 test_result2_mat = [test_result2_mat result2];
 test_result3_mat = [test_result3_mat result3];
 test_result4_mat = [test_result4_mat result4];
 test_result5_mat = [test_result5_mat result5];
 t = zeros(testDoFlim,1);
 t(testDoF) = 1;
 testAns = [testAns t];
end
detectionRate1(indexDoFlim) = sum(sum(test_result1_mat .* testAns)) / nTest
detectionRate2(indexDoFlim) = sum(sum(test_result2_mat .* testAns)) / nTest
detectionRate3(indexDoFlim) = sum(sum(test_result3_mat .* testAns)) / nTest
detectionRate4(indexDoFlim) = sum(sum(test_result4_mat .* testAns)) / nTest
detectionRate5(indexDoFlim) = sum(sum(test_result5_mat .* testAns)) / nTest
end
%% figure part
figure(2)
plot(trainDoFlimset,detectionRate5,'r-o','lineWidth',1)
hold on
plot(trainDoFlimset,detectionRate1,':pentagram','linewidth',1)
hold on
grid on
plot(trainDoFlimset,detectionRate2,':v','linewidth',1)
hold on
plot(trainDoFlimset,detectionRate3,':>','linewidth',1)
hold on
plot(trainDoFlimset, detectionRate4,':<','lineWidth',1)
hold on
xlabel("K_{max}")
xticks([4, 6, 8, 10, 12])
ylabel("Detection rate")
legend("Proposed method","Method using each column component","Method using each row component","Method using each diagonal component","Method using each anti-diagonal component",'location','best');