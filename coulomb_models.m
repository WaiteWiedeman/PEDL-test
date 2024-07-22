close all;
clear;
clc;
params = parameters();
G = params(4);
M1 = params(5);
M2 = params(6);
N = (M1+M2)*G;
mu_s = params(7);
mu_k = params(8);
ctrlOptions = control_options();
v = -5:0.01:5;
flag = "coulomb";
switch flag
    case 'smooth'
        vd = 0.1; % m/s
        fc = mu_s*N*tanh(v/vd);    
    case 'coulomb'       
        vd = 0.01; % m/s
        fc = zeros(1,length(v));
        F = 0;
        for i = 1:length(v)
            if abs(v(i)) < vd
                fc(i) = min(max(F,mu_s*N),mu_s*N);
            else
                fc(i) = mu_k*N*sign(v(i));
            end
        end
    case "Andersson"
        vd = 0.1; % m/s
        p = 2;
        k = 10000;
        fc = zeros(1,length(v));
        for i = 1:length(v)
            fc(i) = N*(mu_k + (mu_s - mu_k)*exp(-(abs(v(i))/vd)^p))*tanh(k*v(i)); 
        end
    case "Specker"
        vd = 0.05; % m/s
        fc = zeros(1,length(v));
        vt = vd*1.5;
        kv = 0;

        for i = 1:length(v)
            fc(i) = (N*mu_s - N*mu_k*tanh(vt/vd) - kv*vt)*(v(i)/vt)*...
                exp(0.5 - 0.5*(v(i)/vt)^2) + N*mu_k*tanh(v(i)/vd); 
        end
end    

figure('Position',[500,100,800,800]);
plot(v,fc,'b-','LineWidth',2);
