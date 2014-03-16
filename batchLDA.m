function [gamma, lambda] = batchLDA(wc, varargin)
% BATCHLDA Batch Variational Bayes for Latent Dirichlet Allocation
% gamma = BATCHLDA(wc) learns a DxK matrix of topic features gamma
% from a DxW matrix of document word count vectors wc. D is the size
% of the corpus and W is the size of the vocabulary. The topic features
% can be input into a classifier for text categorization.
%
% The default number of topics is 50. To override this value, use:
% gamma = BATCHLDA(..., 'K', 40)
%
% [gamma, lambda] = BATCHLDA(...) additionally returns the KxW matrix of
% vocabulary parameters lambda.

% Christopher Sang
% 2013-12-11

p = inputParser;
addRequired(p, 'wc', @isnumeric);
addParamValue(p, 'alpha', 0.01, @isnumeric);
addParamValue(p, 'eta', 0.01, @isnumeric);
addParamValue(p, 'K', 50, @(x) isnumeric(x) && x > 0);
addParamValue(p, 'epsilon', 0.00001, @isnumeric);
addParamValue(p, 'initLambda', [], @isnumeric);
addParamValue(p, 'maxIter', 250, @isnumeric);
addParamValue(p, 'eStepMaxIter', 250, @isnumeric);
addParamValue(p, 'verbose', false, @islogical);
parse(p, wc, varargin{:});


%% Parameters

% model parameters (constant for now, can be adaptively set later)
alpha = p.Results.alpha;
eta = p.Results.eta;

% number of topics
K = p.Results.K;

% corpus size and vocab size, inferred from input dataset
[D, W] = size(wc);

% threshold for convergence
epsilon = p.Results.epsilon;

% maximum number of LDA iterations
maxIter = p.Results.maxIter;
eStepMaxIter = p.Results.eStepMaxIter;

verbose = p.Results.verbose;
checkpoint = 25;


%% Initialization

% the "topics" (parameterizes the vocabulary distribution, beta)
% [experimental feature -- resume a scan of a corpus by restoring the value
%  of lambda here]
lambda = p.Results.initLambda;
if isempty(lambda)
    lambda = 0.5 + rand(K, W);
end

% per-document features (parameterizes the topic distribution, theta)
gamma = 0.5 + rand(D, K);

% multinomial probabilities for word generation
%phi = zeros(D, W, K); % (too big to keep in memory -- keep track of phi_d)

% change in topics at each iteration (used to test for convergence)
dLambda = nan(maxIter, 1);


%% LDA Iterations

eStepOK = true;
iter = 1;
while true

    prevLambda = lambda;
    
    sumNTimesPhi = zeros(W, K); % accumulator for optimizing M step
    
    %gamma = rand(D, K); % reset document params for E step
    
    % E step
    for d = 1:D
        
        % optimize phi and gamma holding lambda fixed
        eStepIter = 1;
        while true
            prevGamma_d = gamma(d, :);
            
            % compute expectations of log(theta) and log(beta)
            e1 = psi(gamma(d, :)) - psi(sum(gamma(d, :)));  % 1 x K
            e2 = bsxfun(@minus, psi(lambda), psi(sum(lambda, 2))); % K x W
            e = bsxfun(@plus, e1, e2.'); % W x K
            phi_d = exp(e);
            phi_d = bsxfun(@rdivide, phi_d, sum(phi_d));
            
            phiTimesN = bsxfun(@times, phi_d, wc(d, :).');
            gamma(d, :) = alpha + sum(phiTimesN);
            %phi(d, :, :) = phi_d;
            
            if any(any(isnan(gamma))) || any(any(gamma < 0))
                error('Invalid value for gamma');
            end
            
            dGamma = sum(abs(prevGamma_d - gamma(d, :)), 2) / K;
            if dGamma < epsilon
                break
            end

            if eStepIter >= eStepMaxIter
                if eStepOK
                    warning('E-step iterations did not converge (iter = %d, d = %d)\n-- this warning will only be shown once', iter, d);
                    eStepOK = false;
                end
                break
            end
            eStepIter = eStepIter + 1;
        end
        
        sumNTimesPhi = sumNTimesPhi + phiTimesN;
    end
    
    % M step
    lambda = eta + sumNTimesPhi.';
    
    % check for LDA convergence
    dLambda(iter) = sum(sum(abs(lambda - prevLambda)));
    if dLambda(iter) < epsilon
        if verbose
            fprintf('LDA converged after %d iterations\n', iter);
        end
        break
    end
    
    if iter >= maxIter
        warning('LDA iterations did not converge (dLambda = %g)', dLambda(iter));
        break
    end
    
    % print progress indicator
    if verbose && mod(iter, checkpoint) == 0
        fprintf('iter = %d, dLambda = %g\n', iter, dLambda(iter));
    end
    
    iter = iter + 1;
end

end % function

