data {
  int<lower = 1> N;
  int<lower = 1> K;
  int<lower = 1> p;
  matrix[N, K] x;
  int<lower = 0,upper=50000> y[N];
  vector[N] y_real;
  matrix[N, p] Q;
  vector[p] lambdas;
}

transformed data {
  vector[p] log_lambdas = log(lambdas);
  real sum_lambdas = sum(lambdas);

  // Compute, thin, and then scale QR decomposition
  matrix[N, K] X_Q = qr_Q(x)[, 1:K] * N;
  matrix[K, K] X_R = qr_R(x)[1:K, ] / N;
  matrix[K, K] X_R_inv = inverse(X_R);
}

parameters {
  real beta0;                       // intercept
  vector[K] betas_tilde;            // coefficients
  real<lower = 0> tau;              //scale for Ridge Regression
  real<lower = 0, upper=2> v;       //scale for Ridge Regression
  vector[p] phi;                    // spatial effects
  real<lower = 0.001> phi2_reciprocal;
  real<lower = 0> inv_alpha;
  real<lower = 0> inv_tau2;
}

transformed parameters {
  // Var(X) = mu + mu^2 / phi
  // phi -> infty no dispersion, phi -> 0 is full dispersion
  // this is 1 / phi, so that phi_reciprocal = 0 is no dispersion
  real phi2 = 1. / sqrt(phi2_reciprocal);
  vector[p] lambdas2;
  vector[K] betas;
  real<lower = 0> alpha = 1./inv_alpha;
  real<lower = 0> tau2 = 1./inv_tau2; //scale for Ridge Regression
  vector<lower = 0>[p] phi_tau;

  real intercept = beta0;
  vector[N] fixedE;
  vector[N] randomE;
  vector[N] mu_nb;

  lambdas2 =  softmax(alpha*log_lambdas);
  lambdas2 = sum_lambdas*lambdas2;
  phi_tau = tau2 * sqrt(lambdas2);

  betas = X_R_inv * betas_tilde;

  fixedE = X_Q * betas_tilde;
  randomE = Q*phi;

  mu_nb = intercept + fixedE + randomE;
}


model {
  beta0 ~ normal(0.0, 1.0);
  betas ~ normal(0.0, tau);
  tau ~ cauchy(0,1);

  phi2_reciprocal ~ cauchy(0., 2.);

  phi ~ normal(0, phi_tau);
  inv_alpha ~ gamma(2, 5);
  inv_tau2 ~ gamma(v/2, v/2);
  v ~ gamma(2, 0.1);

  y ~ neg_binomial_2_log(mu_nb, phi2);

  // soft sum-to-zero constraint on phi)
  sum(randomE) ~ normal(0, 0.01 * N);  // equivalent to mean(phi) ~ normal(0, 0.01)
}

generated quantities {
    vector[N] log_lik;
    int<lower=0> y_pred_full[N];
    int<lower=0> y_pred[N];
    vector[N] mu_nb2 = intercept + fixedE;
    real varI = log(1 + 1 / mean(y_real) + 1 / phi2);
    real varF = variance(fixedE);
    real varR = variance(randomE);
    real r2 = varF / (varF + varR + varI);
    real r2_c = (varF + varR) / (varF + varR + varI);

    int overflow = 0;
    for (i in 1:N) {
        y_pred_full[i] = 99999;
        y_pred[i] = 99999;
        if(mu_nb2[i] <= 0){
            mu_nb2[i] = 0.0000000001;
        }

        if(mu_nb[i] > 10) {
            log_lik[i] = neg_binomial_2_log_lpmf(y[i] | 9, phi2);
            overflow = 1;
        } else {
            log_lik[i] = neg_binomial_2_log_lpmf(y[i] | mu_nb[i], phi2);
        }
    }

    if (overflow == 0){
        y_pred = neg_binomial_2_log_rng(mu_nb2, phi2);
        y_pred_full = neg_binomial_2_log_rng(mu_nb, phi2);
    }
}