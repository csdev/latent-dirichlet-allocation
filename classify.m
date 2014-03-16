% LDA Example using PRT Classifiers

%%

clearvars -except ds
[gamma, lambda] = batchLDA(ds.getX(), 'K', 10, 'verbose', true);
dsTopics = prtDataSetClass(gamma, ds.getY());
dsTopics = dsTopics.setClassNames(ds.getClassNames());


%%

zmuv = prtPreProcZmuv;

class = prtClassRvm;
class.internalDecider = prtDecisionBinaryMinPe;

alg = zmuv + class;

% single training/testing split
% keys = dsTopics.getKFoldKeys(10);
% dsTrain = dsTopics.retainObservations(keys == 1);
% dsTest = dsTopics.retainObservations(keys ~= 1);
% 
% alg = alg.train(dsTrain);
% result = alg.run(dsTest);

% full cross validation
result = alg.kfolds(dsTopics, 5);

figure;
prtScoreConfusionMatrix(result);
