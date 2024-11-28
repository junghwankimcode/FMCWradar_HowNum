clear all
clc

%% parameter setting part
nTrain = 1e2;

%str = 12;
trainDoFlim = 4;

SNRmargin = 0;

Scene = 2;

% radar parameters
f_c = 76.5 * 10^9;
c = 3 * 10^8;
lambda = c / f_c;
BW = 30e6;
Tsweep = 10e-6;
num_C = 16;
f_s = 2.4e6;
T_s = 1 / f_s;
num_T = fix(Tsweep / T_s);
K = BW / Tsweep;

% determined capabilities
R_max = f_s * Tsweep * c / (2 * BW);
V_max = c / (2 * (f_c * Tsweep));

trainSNRset = [20];

%% number of signal received targers estimation part
trainData1_row = [];
trainData1_col =[];
trainData1_abs=[];
trainData1_fft=[];
labels = [];

for indexSNR = 1:length(trainSNRset)

    trainSNR = trainSNRset(indexSNR);

    % Training Phase
    parfor indexN = 1:nTrain
        trainDoF = randi(trainDoFlim, 1); 
        gain = (randn(1, trainDoF) + 1i * randn(1, trainDoF)) / sqrt(2);
        R = R_max * rand(1, trainDoFlim);
        V = V_max * rand(1, trainDoFlim);
        [~, ~, observation] = gsGen_2DFMCW(Scene, trainSNR, trainDoF, num_T, num_C, gain, 2 * R, V, c, f_c, f_s, Tsweep, K);

        % Case 2 : Row vector
        xr = observation;
        % for i = 1:num_T
        %     data = xr(i, :);
        %     [~, Sn, ~] = makeHankel(data);
        %     Sn = diag(Sn);
        %     trainData1_row = [trainData1_row Sn];
        %     trainData1_abs=[trainData1_abs abs(data).'];
        %     trainData1_fft=[trainData1_fft abs(fft(data)).'];
        %     labels = [labels; trainDoF];
        % end

        for i = 1:num_C
            data = xr(:,i);
            [~,Sn,~] = makeHankel(data.');
            Sn = diag(Sn);
            trainData1_col = [trainData1_col Sn];
            atrainData1_abs=[trainData1_abs abs(data).'];
            trainData1_fft=[trainData1_fft abs(fft(data)).'];
            labels = [labels; trainDoF];
        end

    end
end

% 데이터가 2차원으로 변환될 수 있도록 행렬 형태로 변환
data1 = trainData1_col'; % t-SNE에 입력할 데이터
data2 = trainData1_abs'; %abs
data3 = trainData1_fft'; %fft

% 각 클래스에 속하는 데이터 개수 확인 (디버깅)
disp('Class distribution:');
for i = 1:trainDoFlim
    disp(['Class ' num2str(i) ': ' num2str(sum(labels == i))]);
end

% t-SNE 적용
Y1 = tsne(data1);
Y2 = tsne(data2);
Y3 = tsne(data3);

% 각 클래스에 대한 색상 설정
colors = lines(trainDoFlim);

%% t-SNE 결과를 플롯
figure(1);

hold on;
for i = 1:trainDoFlim
    scatter(Y1(labels == i, 1), Y1(labels == i, 2), 15, colors(i, :), 'filled');
end
hold off;

% 그래프 설정
title('hankel');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
legend(arrayfun(@(x) ['K= ' num2str(x)], 1:trainDoFlim, 'UniformOutput', false));
grid on;

figure(2);
hold on;
for i = 1:trainDoFlim
    scatter(Y2(labels == i, 1), Y2(labels == i, 2), 15, colors(i, :), 'filled');
end
hold off;

% 그래프 설정
title('abs');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
legend(arrayfun(@(x) ['K=  ' num2str(x)], 1:trainDoFlim, 'UniformOutput', false));
grid on;


figure(3);
hold on;
for i = 1:trainDoFlim
    scatter(Y1(labels == i, 1), Y1(labels == i, 2), 15, colors(i, :), 'filled');
end
hold off;

% 그래프 설정
title('fft');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
legend(arrayfun(@(x) ['K=  ' num2str(x)], 1:trainDoFlim, 'UniformOutput', false));
grid on;


