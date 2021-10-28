clc
clear all

n_elem = 600;
Ts = 1

F = zeros(3);
I3 = eye(3);

Phi = I3;

Q = 0.0001*I3;
Qd = Ts*Q;

p_k = [...
    100 100 -100;
    -100 100 100;
    100 -100 -100;
    -100 100 100;
    300 200 -300;
    -30 200 300;
    200 -300 -200;
    -200 300 200;
        300 200 300;
    -300 200 -300;
    200 -300 200;
    -200 300 -200;
    ]';

p = [50 50 0]'

n_beacons = size(p_k,2)


p0      = [0;0;0];
P0      = 1*eye(3);
r       = 1;


%% init vec

P       = P0;
p_hat   = p0;

tic 
for j = 1:n_elem
    z       = [];
    z_hat   = [];
    H       = [];
    R = [];
    for k = 1:n_beacons
        z           = [ z; norm( p - p_k(:,k) ) ];
        z_hat_temp  = norm( p_hat - p_k(:,k) );
        z_hat       = [ z_hat; z_hat_temp; ];
        H           = [H; (p_hat - p_k(:,k))'/z_hat_temp];
        H
        R           = blkdiag( R, r);
    end
    K = P*H'/(H*P*H' + R);
    d_p  = K*(z-z_hat);
    p_hat = p_hat + d_p;
    P_j = (I3-K*H);
    P = P_j*P*P_j' + K*R*K';
    
    P = Phi*P*Phi' + Qd;
    P = (P+P')/2;
end
t1 = toc;
fprintf('Elapsed time vector impl %f\n',t1);
p_hat
est_err_p = p-p_hat

disp('----')

%% init seq

p0      = [0;0;0];
P0      = 1*eye(3);
r       = 1;

tic 
for j = 1:n_elem
    z       = [];
    z_hat   = [];
    H       = [];
    R_temp  = [];

    d_p = zeros(3,1);
    for k = 1:n_beacons

        z_hat_temp  = norm(p_hat - p_k(:,k));
        H           = (p_hat - p_k(:,k))'/z_hat_temp;
     
        z_hat       = norm( p_hat - p_k(:,k) ) + H*d_p;
        H*d_p
        z           = norm( p - p_k(:,k) );

        R           = r;
        K           = P*H'/(H*P*H' + R);
        d_p
        z-z_hat
        K*(z-z_hat)
        d_p         = d_p + K*(z-z_hat);
        d_p
        z_hat
        z_hat_temp
        z_hat

%         z_hat       = norm(p_hat + d_p - p_k(:,k));
%         H           = (p_hat + d_p - p_k(:,k))'/z_hat;
%         z           = norm( p - p_k(:,k) );
%         R           = r;
%         K           = P*H'/(H*P*H' + R);
%         d_p         = d_p + K*(z-z_hat);

        P_j         = (I3-K*H);
        P           = P_j*P*P_j' + K*R*K';
%         R_temp      = blkdiag( R_temp, r); % selv med denne unødvendig linja er seq impl. mye raskere. Tok bare med denne linja siden jeg vet at blkdiag er en "dyr" funksjon å kalle
    end
    p_hat = p_hat + d_p;

    
    P = Phi*P*Phi' + Qd;
    P = (P+P')/2;
end
t2 = toc;
fprintf('Elapsed time seq impl %f\n',t2);
p_hat
est_err_p = p-p_hat

speed_up = (t1-t2)/t1*100;
fprintf('Speedup time seq impl %f per cent \n', speed_up);

