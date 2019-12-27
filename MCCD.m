% CORRESPONDENCE INFORMATION
%    This code is written by Hong Tao, National University of Defense Technology, Changsha, China, 410072

%   WORK SETTING:
%    This code has been compiled and tested by using matlab 2016a

%  For more detials, please see the manuscript:
%   Hong Tao, Chenping Hou, Dongyun Yi,and Jubo Zhu. 
%   Multi-View Classification with Cohesion and Diversity.
%   IEEE Transactions on Cybernetics (T-CYB).

%   Last Modified: 2018.09


function [res, W,b, M,obj ,ypre] = MCCD(Xtr,Xte, Ytr, Yte, gamma1,gamma2, maxIter  )
% self-weighted multi-view multi-class SVM
% X: cell, data matrix, nSmp*nfea
% Y: \in {0,+1}, Y(i,j) = 1 or 0 indicates that xi belongs to the j-th
%    class or not
% gamma: regularization parameter
if ~exist('maxIter','var')
    maxIter = 50;
end
viewNum = length(Xtr);
[nSmp,nClass] = size(Ytr);
Dim = zeros(viewNum,1);
for v = 1:viewNum
    Dim(v) = size(Xtr{v},2);
end
In = eye(nSmp);
en = ones(nSmp,1);
% Enc = ones(nSmp,nClass);
% prepare B 
B = -ones(nSmp,nClass);
for i = 1:nClass
    B(Ytr(:,i) == 1,i) = 1;
end
[~,gndtr] = max(Ytr,[],2);
 [~,gndte] = max(Yte,[],2);
% intitialize M as zero matrix
M = zeros(nSmp,nClass);

% the allowed perturbation of objective value
threshold = 1e-6;
% the small positive value used to avoid zero in the dominatant when
% updating u
epsilon = 1e-8;

% initialize W and b for each view
W = cell(viewNum,1); b = W;
if numel(gamma1)  == 1
    gamma1 = ones(viewNum,1)*gamma1;
end
for v = 1:viewNum
    [W{v}, b{v}] = least_squares_regression(Xtr{v}',  Ytr',  gamma1(v));
end
% initialize U for each view;
u = cell(viewNum,1);
for v = 1:viewNum
    temp = Xtr{v}*W{v} + en*b{v}' - Ytr - B.*M;
    temp = sum(temp.*temp,2);
    u{v} = 0.5./sqrt(temp + epsilon);
end

Hn = eye(nSmp) - ones(nSmp)/nSmp;

% begin the optimization
obj = [];
for iter = 1:maxIter
    % calculate W and b
    Z = Ytr + B.*M;
    sumU = zeros(nSmp);
    sumQ = zeros(nSmp,nClass);
    for v = 1:viewNum
        sumKHn = zeros(nSmp);
        for v1 = 1:viewNum
            if v1 == v
                sumKHn = sumKHn + 0;
            else
                sumKHn =sumKHn + Xtr{v1}*W{v1}*W{v1}'*Xtr{v1}';
            end
        end
        sumKHn = Hn*sumKHn*Hn;
        K = Xtr{v}'*sumKHn*Xtr{v};
        Uv = diag(u{v});
        H = Uv-1/sum(u{v})*Uv*en*en'*Uv;

        G = Xtr{v}'*H*Xtr{v}+gamma1(v)*eye(Dim(v)) + gamma2*K;
        W{v} = G\(Xtr{v}'*H*Z);
        b{v} = (Z'-W{v}'*Xtr{v}')*Uv*en/sum(u{v});
        sumU = sumU + Uv;
        sumQ = sumQ + Uv*(B.*(Xtr{v}*W{v}+en*b{v}'-Ytr));
        clear G H Uv
    end
   
    % update M
    M = max(diag(1./(diag(sumU)+epsilon))*sumQ,0);
    
    % update U and calculate the objective value
    obj(iter) = 0;
    for v = 1:viewNum
        temp = Xtr{v}*W{v} + en*b{v}' - Ytr - B.*M;
        temp = sum(temp.*temp,2);
        u{v} = 0.5./sqrt(temp +epsilon);
        sumKHn = zeros(nSmp);
        for v1 = 1:viewNum
            if v1 == v
                sumKHn = sumKHn + 0;
            else
                sumKHn =sumKHn + Xtr{v1}*W{v1}*W{v1}'*Xtr{v1}';
            end
        end
        obj(iter) =obj(iter) + sum(sqrt(temp)) + gamma1(v) * sum(sum(W{v}.*W{v})) + gamma2*trace(W{v}'*Xtr{v}'*Hn*sumKHn*Hn*Xtr{v}*W{v});
    end
    

    if iter > 6 && (obj(iter)-obj(iter-1))/obj(iter-1) < threshold
        break;
    end
end

%% classifying the testing points

[res,ypre,F]=make_pred(Xte,gndte,W,b);
res

end
function [res,ypre,F]=make_pred(Xte,gndte,W,b)
nClass = length(b{1});
Ic = eye(nClass);
nteSmp = size(Xte{1},1);
en = ones(nteSmp,1);
F = zeros(nteSmp,nClass);
viewNum = length(Xte);
for cc = 1:nClass
    Ycc = repmat(Ic(cc,:),nteSmp,1);
    for v = 1:viewNum
        temp = Xte{v}*W{v} + en*b{v}' - Ycc;
        F(:,cc) = F(:,cc) + sqrt(sum(temp.*temp,2));
    end
end
[~,ypre] = min(F,[],2);
Acc = sum(ypre == gndte)/nteSmp;
F1 = macroFbeta(ypre,gndte);
res = [Acc,F1];
end

