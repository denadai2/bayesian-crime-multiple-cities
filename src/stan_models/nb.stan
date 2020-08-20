data {
  int<lower = 1> N;
  int<lower = 1> K;
  matrix[N,K] x;
  int<lower = 0,upper=50000> y[N];
  vector[N] y_real;
}

transformed data {
  // Compute, thin, and then scale QR decomposition
  matrix[N, K] X_Q = qr_Q(x)[, 1:K] * N;
  matrix[K, K] X_R = qr_R(x)[1:K, ] / N;
  matrix[K, K] X_R_inv = inverse(X_R);
}


parameters {
  real beta0;            // intercept
  vector[K] betas_tilde;      // coefficients
  real<lower = 0.001> phi2_reciprocal;
  real<lower = 0> tau; //scale for Ridge Regression
}

transformed parameters {
  real<lower = 0.001> phi2;
  vector[K] betas = X_R_inv * betas_tilde;
  vector<upper=500000>[N] mu_nb;

  // Var(X) = mu + mu^2 / phi
  // phi -> infty no dispersion, phi -> 0 is full dispersion
  // this is 1 / phi, so that phi_reciprocal = 0 is no dispersion
  phi2 = 1. / (phi2_reciprocal) + 0.001;

  mu_nb = beta0 + X_Q * betas_tilde;
}

model {
  beta0 ~ normal(0.0, 1.0);
  betas ~ normal(0.0, tau);
  tau ~ cauchy(0,1);
  phi2_reciprocal ~ cauchy(0., 2.);
  y ~ neg_binomial_2_log(mu_nb, phi2);
}

generated quantities {
    vector[N] log_lik;
    vector[N] llnoRE;
    int<lower=0> y_pred[N];
    int<lower=0> y_pred_full[N];
    vector[N] randomE;
    vector[N] fixedE = X_Q * betas_tilde;
    real intercept = beta0;
    vector[N] eta = intercept + fixedE;
    vector[N] mu = exp(eta);
    real varI = log(1. + 1. / mean(y_real) + 1. / phi2);
    real varF = variance(fixedE);
    real varR = 0;
    real r2 = varF / (varF + varR + varI);
    real r2_c = (varF + varR) / (varF + varR + varI);


    for (i in 1:N) {
        if(mu[i] > 10000 || mu[i] <= 0) {
            y_pred[i] = 99999;
            y_pred_full[i] = 99999;
            log_lik[i] = neg_binomial_2_lpmf(y[i] | 99999, phi2);
            llnoRE[i] = neg_binomial_2_lpmf(y[i] | 99999, phi2);
        } else {
            y_pred[i] =  neg_binomial_2_rng(mu[i], phi2);
            y_pred_full[i] = y_pred[i];
            log_lik[i] = neg_binomial_2_lpmf(y[i] | mu[i], phi2);
            llnoRE[i] = neg_binomial_2_lpmf(y[i] | mu[i], phi2);
        }
        randomE[i] = 0;

    }
}