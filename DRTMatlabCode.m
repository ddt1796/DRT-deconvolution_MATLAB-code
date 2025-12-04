clc;
close all;

%% 1. INPUT YOUR EIS DATA HERE
% The matrix must have 3 columns:
% Column 1: Frequency (in Hz)
% Column 2: Z' (Real Impedance)
% Column 3: -Z'' (Negative Imaginary Impedance)

if ~exist('Z', 'var')
    error('Data matrix "Z" not found in workspace. Please create it before running the script. It should have 3 columns [Frequency, Z_real, -Z_imag].');
elseif size(Z, 2) < 3
    error('Data matrix "Z" must have 3 columns: [Frequency, Z_real, -Z_imag].');
end

freq_exp = Z(:, 1);       % Experimental frequency data (Hz)
Z_real_exp = Z(:, 2);     % Z' experimental data (Ohm)
Z_imag_neg_exp = Z(:, 3); % -Z'' experimental data (Ohm)

% Combine Z' and -Z'' into the measured data vector (Z_real, -Z_imag)
Z_exp_vector = [Z_real_exp; Z_imag_neg_exp];

%% 2. PARAMETER SETUP & TIKHONOV REGULARIZATION
% 2.1. Tikhonov Regularization Parameter (Lambda)
lambda = 0.01; 

% 2.2. Relaxation Time (tau) Grid Setup
% Number of discrete tau points (resolution of the DRT plot)
M = 100; 

% Define the range of tau (s)
log_tau_min = log10(1 / (2 * pi * max(freq_exp)));
log_tau_max = log10(1 / (2 * pi * min(freq_exp)));

log_tau = linspace(log_tau_min - 1, log_tau_max + 1, M)';
tau = 10.^log_tau;
d_log_tau = log_tau(2) - log_tau(1);

% 2.3. Estimate Rs and Remove from Data
Rs_est = Z_real_exp(1); 
Z_real_prime = Z_real_exp - Rs_est;
Z_exp_vector_prime = [Z_real_prime; Z_imag_neg_exp];

%% 3. CONSTRUCT THE DRT KERNEL MATRIX (K)
% The DRT equation: Z(f) = Rs + Integral [ G(tau) / (1 + i*2*pi*f*tau) ] d(log tau)

N = length(freq_exp);
omega = 2 * pi * freq_exp;

% Initialize Kernel K for Real and Negative Imaginary parts
K_real = zeros(N, M);
K_imag = zeros(N, M);

for m = 1:M
    % The complex kernel for the impedance Z_RC
    Complex_Kernel = 1 ./ (1 + 1i * omega * tau(m));
    
    % Store the real and negative imaginary parts of the kernel
    K_real(:, m) = real(Complex_Kernel) * d_log_tau;
    K_imag(:, m) = -imag(Complex_Kernel) * d_log_tau; 
end

K = [K_real; K_imag];


%% 4. TIKHONOV REGULARIZATION & SOLUTION
% First-order Tikhonov regularization (L is the discrete derivative operator)
% Regularized linear system: min(||K*G - Z_prime||^2 + lambda^2 * ||L*G||^2)
% The combined problem is: min(|| [K; lambda*L] * G - [Z_prime; 0] ||^2)

% Construct the first-order difference matrix (L) for smoothing
L = zeros(M, M);
for i = 1:M-1
    L(i, i) = -1;
    L(i, i+1) = 1;
end
% Set the last row to 0 for stability
L(M, M) = 0; 

% Construct the Tikhonov matrix components
K_tilde = [K; lambda * L];
Z_tilde = [Z_exp_vector_prime; zeros(M, 1)];

% Solve for G(tau) using non-negative least squares (G(tau) >= 0)
G_tau_fit = lsqnonneg(K_tilde, Z_tilde);

%% 5. VALIDATION: CALCULATE FITTED IMPEDANCE

Z_fit_vector_prime = K * G_tau_fit;
Z_real_fit = Z_fit_vector_prime(1:N) + Rs_est;
Z_imag_neg_fit = Z_fit_vector_prime(N+1:end);

%% 6. VISUALIZE THE RESULTS

figure('Name', 'DRT Analysis Results');

% --- Subplot 1: Nyquist Plot Validation ---
subplot(2, 1, 1);
hold on;
grid on;
plot(Z_real_exp, Z_imag_neg_exp, 'bo', 'DisplayName', 'Experimental Data','MarkerFaceColor', 'b');
plot(Z_real_fit, Z_imag_neg_fit, 'r-', 'LineWidth', 3, 'DisplayName', 'DRT Reconstruction');
title('Nyquist Plot: Experimental Data vs. DRT Reconstruction');
xlabel("Z' / Ohm.cm^2");
ylabel("-Z'' / Ohm.cm^2");
legend('show', 'Location', 'best');
axis equal;
hold off;

% --- Subplot 2: Distribution of Relaxation Times G(tau) ---
subplot(2, 1, 2);
hold on;
grid on;
semilogx(log(tau), G_tau_fit, 'k-', 'LineWidth', 2);
title(['Distribution of Relaxation Times G(\tau) (Lambda = ', num2str(lambda), ')']);
xlabel('Relaxation Time, \tau (s)');
ylabel('Distribution Function, G(\tau) (\Omega.cm^2)');
hold off;