clc
clear all
% Cargar archivo CSV como matriz
data = readmatrix('resultadosGeneral1.csv');



M = data(:,[1:17]);

%Clasificador SVM

[vp,vs] = ClassSV(data);
vp.ClassificationSVM;
vp.HowToPredict;

%Clasificador FT

[FTvp,vs] = ClassFT(data);
FTvp.ClassificationTree;
FTvp.HowToPredict;

Y=data(:,18);

[yfit,scores] = FTvp.predictFcn(M);


positive_scores = scores(Y == 1, 2); % Puntuaciones para clase positiva
negative_scores = scores(Y == 0, 2); % Puntuaciones para clase negativa


% Estimar densidades
[f_positive, xi_positive] = ksdensity(positive_scores);
[f_negative, xi_negative] = ksdensity(negative_scores);

% Graficar funciones de densidad
figure (1);
plot(xi_positive, f_positive, 'b', 'LineWidth', 2); hold on;
plot(xi_negative, f_negative, 'r', 'LineWidth', 2);
xlabel('Risk');
ylabel('Density');
legend('Yess', 'No');
title('DT');
grid on;


% [yfit,scoresSVM] =vp.predictFcn(M);
% 
% positive_scores = scoresSVM(Y == 1, 2); % Puntuaciones para clase positiva
% negative_scores = scoresSVM(Y == 0, 2); % Puntuaciones para clase negativa
% 
% 
% % Estimar densidades
% [f_positive, xi_positive] = ksdensity(positive_scores);
% [f_negative, xi_negative] = ksdensity(negative_scores);
% 
% % Graficar funciones de densidad
% figure;
% plot(xi_positive, f_positive, 'b', 'LineWidth', 2); hold on;
% plot(xi_negative, f_negative, 'r', 'LineWidth', 2);
% xlabel('Puntuaciones de decisión');
% ylabel('Densidad');
% legend('Clase Positiva', 'Clase Negativa');
% title('Función de Densidad de Puntuaciones para SVM');
% grid on;



% Calibrar el modelo para obtener probabilidades
[trainedModelWithPosterior, ScoreTransform] = fitPosterior(vp.ClassificationSVM, M, Y);

% Obtener probabilidades
[~, probabilities] = predict(trainedModelWithPosterior, M);

% Repetir el análisis de densidad con probabilidades
positive_probs = probabilities(Y == 1, 2);
negative_probs = probabilities(Y == 0, 2);

[f_positive_prob, xi_positive_prob] = ksdensity(positive_probs);
[f_negative_prob, xi_negative_prob] = ksdensity(negative_probs);

% Graficar densidad de probabilidades
figure (2);
plot(xi_positive_prob, f_positive_prob, 'b', 'LineWidth', 2); hold on;
plot(xi_negative_prob, f_negative_prob, 'r', 'LineWidth', 2);
xlabel('Risk');
ylabel('Density');
legend('Yess', 'No');
title('SVM');
grid on;


%Clasificador  RF

[RFvp,vs] = ClassRF(data);
RFvp.ClassificationEnsemble;
RFvp.HowToPredict;

Y=data(:,18);

[yfit,scores] = RFvp.predictFcn(M);


positive_scores = scores(Y == 1, 2); % Puntuaciones para clase positiva
negative_scores = scores(Y == 0, 2); % Puntuaciones para clase negativa


% Estimar densidades
[f_positive, xi_positive] = ksdensity(positive_scores);
[f_negative, xi_negative] = ksdensity(negative_scores);

% Graficar funciones de densidad
figure (3);
plot(xi_positive, f_positive, 'b', 'LineWidth', 2); hold on;
plot(xi_negative, f_negative, 'r', 'LineWidth', 2);
xlabel('Risk');
ylabel('Density');
legend('Yess', 'No');
title('RF');
grid on;


%Clasificador  KNN

[KNNvp,vs] = ClassKNN(data);
KNNvp.ClassificationKNN;
KNNvp.HowToPredict;

Y=data(:,18);

[yfit,scores] = KNNvp.predictFcn(M);


positive_scores = scores(Y == 1, 2); % Puntuaciones para clase positiva
negative_scores = scores(Y == 0, 2); % Puntuaciones para clase negativa


% Estimar densidades
[f_positive, xi_positive] = ksdensity(positive_scores);
[f_negative, xi_negative] = ksdensity(negative_scores);

% Graficar funciones de densidad
figure (4);
plot(xi_positive, f_positive, 'b', 'LineWidth', 2); hold on;
plot(xi_negative, f_negative, 'r', 'LineWidth', 2);
xlabel('Risk');
ylabel('Density');
legend('Yess', 'No');
title('KNN');
grid on;


%Clasificador LR

[LRvp,vs] = ClassLR(data);
LRvp.ClassificationLinear;
LRvp.HowToPredict;

Y=data(:,18);

[yfit,scores] = LRvp.predictFcn(M);


positive_scores = scores(Y == 1, 2); % Puntuaciones para clase positiva
negative_scores = scores(Y == 0, 2); % Puntuaciones para clase negativa


% Estimar densidades
[f_positive, xi_positive] = ksdensity(positive_scores);
[f_negative, xi_negative] = ksdensity(negative_scores);

% Graficar funciones de densidad
figure (5);
plot(xi_positive, f_positive, 'b', 'LineWidth', 2); hold on;
plot(xi_negative, f_negative, 'r', 'LineWidth', 2);
xlabel('Risk');
ylabel('Density');
legend('Yess', 'No');
title('LR');
grid on;