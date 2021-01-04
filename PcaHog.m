function [phtp, phtf, phtep, phtef, ptp, ptf, ptep, ptef] = PcaHog()
    %% Which Algorithm To Use
    PCA_HOG = 1; 
    Only_PCA = 1;
    
    %% Load the datasets
    load('X1600.mat');
    load('Te28.mat');
    load('Lte28.mat');
    
    if(PCA_HOG == 1)
        %% Prepare the datasets
        u = ones(1,1600);
        ytr = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u];
        Dtr = [X1600; ytr];
        Dte = [Te28; 1+Lte28(:)'];
                
        t = cputime;
        H = [];
        for i = 1:16000
         xi = X1600(:,i);
         mi = reshape(xi,28,28);
         hi = hog20(mi,7,9);
         H = [H hi];
        end
        Dhtr = [H; ytr];
        
        [rows,cols] = size(H);
        Xtr = zeros(rows,cols);
        
        %% Mean Centred Data
        for n = 1:rows
            Xtr(n,:) =  (H(n,:) - (1/cols) * sum(H(n,:)));
        end
        
        %% Covariance of the Matrix
        C_m = (1/rows) * Xtr * Xtr';
        
        %% Eigen Vectors and Eigen Values of the Covariance matrix
        [C,c] = eigs(C_m,23);
        
        %% Principal Components
        pcs = (C' * Xtr);
        
        pht = cputime - t;
        
        Dhtr = [pcs; ytr];
        
        %% Create a classifier
        
        y = [-ones(1,16000)]; %create a vector of negative 1's
        K = 100; %Number of iterations
        wh = [];
        m = 1;
        for j = 1:10
        yn = y;
        n = 1600 * j;
        yn(1,m:n) = 1; % assign 1 to positions where you have class Cj
        m = n;
        [ws,C2] = LRBC_newton(pcs,yn,K);
        nw = norm(ws); %calculate norm of wj Note ws = [ wj b]'
        whj = ws/nw; %perform normalization
        wh(:,j) = whj; %store the normalized value
        end
        %% Classify the samples
        Xtr_h = [ pcs; ones(1,16000)]; % add a row of ones so that we create x hat for each input x.
        
        t = cputime;
        [~,ypred_train] = max((Xtr_h' * wh)'); 
        phtp = cputime - t;
        
        phtf = (pht + phtp)/16000
        
        Ctrain = confusionmat(ypred_train,ytr) %confusion matrix for training set
        train_acc = sum(diag(Ctrain))/sum(sum(Ctrain)) %classification accuracy for training set
        
        %% For Testing Data
        t = cputime;
        Hte = [];
        for i = 1:length(Lte28)
         xi = Te28(:,i);
         mi = reshape(xi,28,28);
         hi = hog20(mi,7,9);
         Hte = [Hte hi];
        end
        Dhte = [Hte; 1+Lte28(:)'];
        
        
        [rows,cols] = size(Hte);
        Xte = zeros(rows,cols);
        
        for n = 1:rows
            Xte(n,:) =  (Hte(n,:) - (1/cols) * sum(Hte(n,:)));
        end
        
        pcste = (C' * Xte);
        phte = cputime - t;
        %Dhtr = [pcs; ytr];
        
        Xte_h = [ pcste; ones(1,10000) ];
        y_test = 1+Lte28(:)';
        
        t = cputime;
        [~,ypred_test] = max((Xte_h' * wh)'); %Use equation E2.6
        phtep = cputime - t;
        
        phtef = (phte + phtep)/10000;
        
        Ctest = confusionmat(ypred_test,y_test) %confusion matrix for testing set
        test_acc = sum(diag(Ctest))/sum(sum(Ctest)) %classification accuracy for testing set
    end
    
    if(Only_PCA == 1)
        %% Prepare the datasets   
        u = ones(1,1600);
        ytr = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u];
   
        [rows,cols] = size(X1600);
        Xtr = zeros(rows,cols);
        
        %% Mean Centred Data
        t = cputime();
        for n = 1:rows
            Xtr(n,:) =  (X1600(n,:) - (1/cols) * sum(X1600(n,:)));
        end
        
        %% Covariance of the Matrix
        C_m = (1/cols) * Xtr * Xtr';
        
        %% Eigen Vectors and Eigen Values of the Covariance matrix
        [C,c] = eigs(C_m,25);
        
        %% Transform or Project the data
        pcs = (C' * Xtr);
        pt = cputime - t
        
        Dhtr = [pcs; ytr];
       
        %% Multiclass Classification
        y = [-ones(1,16000)]; %create a vector of negative 1's
        K = 150; %Number of iterations
        wh = [];
        m = 1;
        for j = 1:10
        yn = y;
        n = 1600 * j;
        yn(1,m:n) = 1; % assign 1 to positions where you have class Cj
        m = n;
        [ws,C2] = LRBC_newton(pcs,yn,K);
        nw = norm(ws); %calculate norm of wj Note ws = [ wj b]'
        whj = ws/nw; %perform normalization
        wh(:,j) = whj; %store the normalized value
        end
        %% Classify Training Samples
        Xtr_h = [ pcs; ones(1,16000)]; % add a row of ones so that we create x hat for each input x.
    
        t = cputime;
        [~,ypred_train] = max((Xtr_h' * wh)'); %Use equation E2.6
        ptp = cputime - t
        
        ptf = (ptp+pt)/16000
        
        Ctrain = confusionmat(ypred_train,ytr) %confusion matrix for training set
        train_acc = sum(diag(Ctrain))/sum(sum(Ctrain)) %classification accuracy for training set
        
        %% For Testing Data
        
        [rows,cols] = size(Te28);
        Xte = zeros(rows,cols);
        
        t = cputime;
        for n = 1:rows
            Xte(n,:) =  (Te28(n,:) - (1/cols) * sum(Te28(n,:)));
        end
        
        pcste = (C' * Xte);
        ptep = cputime - t;
        
        Xte_h = [ pcste; ones(1,10000)];
        
        y_test = 1+Lte28(:)';
        
        t = cputime;
        [~,ypred_test] = max((Xte_h' * wh)'); % Classify the samples
        tst = cputime - t;
        
        ptef = (ptep + tst)/10000
        Ctest = confusionmat(ypred_test,y_test) %confusion matrix for testing set
        test_acc = sum(diag(Ctest))/sum(sum(Ctest)) %classification accuracy for testing set
        ptf
    end
end
function f = f_LRBC(w,X)
P = size(X,2);
f = sum(log(1+exp(-X'*w)))/P;
end
function g = g_LRBC(w,X)
P = size(X,2);
q1 = exp(X'*w);
q = 1./(1+q1);
g = -(X*q)/P;
end
function H = h_LRBC(w,X)
[N1,P] = size(X);
q1 = exp(X'*w);
q = q1./((1+q1).^2);
H = zeros(N1,N1);
for p = 1:P
    xp = X(:,p);
    H = H + q(p)*(xp*xp');
end
H = H/P;
end
function [ws,C2] = LRBC_newton(X,y,K)
y = y(:)';
[N,P] = size(X);
N1 = N + 1;
Ine = 1e-10*eye(N1);
ind = 1:1:P;
indp = find(y > 0);
indn = setdiff(ind,indp);
Xh = [X; ones(1,P)];
Xp = Xh(:,indp);
Xn = Xh(:,indn);
Xw = [Xp -Xn];
P1 = length(indp);
k = 0;
wk = zeros(N1,1);
while k < K
  gk = feval('g_LRBC',wk,Xw);
  Hk = feval('h_LRBC',wk,Xw) + Ine;
  dk = -Hk\gk;
  ak = bt_lsearch2019(wk,dk,'f_LRBC','g_LRBC',Xw);
  wk = wk + ak*dk;
  k = k + 1;
end
ws = wk;
yt = sign(ws'*[Xp Xn]);
er = abs(y-yt)/2;
erp = sum(er(1:P1));
ern = sum(er(P1+1:P));
C2 = [P1-erp ern; erp P-P1-ern];
end
function a = bt_lsearch2019(x,d,fname,gname,p1,p2)
rho = 0.1;
gma = 0.5;
x = x(:);
d = d(:);
a = 1;
xw = x + a*d;
parameterstring ='';
if nargin == 5
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
end
if nargin == 6
   if ischar(p1)
      eval([p1 ';']);
   else
      parameterstring = ',p1';
   end
   if ischar(p2)
      eval([p2 ';']);
   else
      parameterstring = ',p1,p2';
   end
end
eval(['f0 = ' fname '(x' parameterstring ');']);
eval(['g0 = ' gname '(x' parameterstring ');']);
eval(['f1 = ' fname '(xw' parameterstring ');']);
t0 = rho*(g0'*d);
f2 = f0 + a*t0;
er = f1 - f2;
while er > 0
     a = gma*a;
     xw = x + a*d;
     eval(['f1 = ' fname '(xw' parameterstring ');']);
     f2 = f0 + a*t0;
     er = f1 - f2;
end
if a < 1e-5
   a = min([1e-5, 0.1/norm(d)]); 
end 
end