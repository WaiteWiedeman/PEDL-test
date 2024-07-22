function fc = coulomb_friction(v, F)
    params = parameters();
    G = params(4);
    M1 = params(5);
    M2 = params(6);
    N = (M1+M2)*G;
    mu_s = params(7);  
    mu_k = params(8);
    ctrlOptions = control_options();
    switch ctrlOptions.friction
        case "none"
            fc = 0;
        case 'smooth'
            vd = 0.01; % m/s
            fc = mu_k*N*tanh(v/vd);    
        case 'coulomb'       
            tolerance = params(9);
            if abs(v) < tolerance
                fc = min(F,mu_s*N);
            else
                fc = mu_k*N*sign(v);
            end
        case "andersson"
            vd = 0.1; % m/s
            p = 2;
            k = 10000;
            fc = N*(mu_k + (mu_s - mu_k)*exp(-(v/vd)^p))*tanh(k*v);
        case "Specker"
            vd = 0.05; % m/s
            vt = vd*2;
            kv = 0;
            fc = (N*mu_s - N*mu_k*tanh(vt/vd) - kv*vt)*(v/vt)*...
                    exp(0.5 - 0.5*(v/vt)^2) + N*mu_k*tanh(v/vd);
        otherwise
            fc = min(1,max(-1,1000*v))*mu_s*N;
    end
end