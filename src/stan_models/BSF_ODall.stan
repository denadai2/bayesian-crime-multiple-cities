functions {
    // From: https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
    real BSF_lpdf(vector phi, real tau,
    matrix Sigma, int p) {
      return 0.5 * (p * log(tau) - tau * (phi' * Sigma * phi));
  }
}

data {
  int<lower = 1> N;
  int<lower = 1> K;
  int<lower = 1> p1;
  int<lower = 1> p2;
  matrix[N, K] x;
  int<lower = 0> y[N];
  vector[N] y_real;
  matrix[N, p1] Q1;
  matrix[N, p2] Q2;
  matrix[N, N] W1;
  matrix[N, N] W2;
}

transformed data {
    matrix[p1, p1] magic1 = (Q1' * W1 * Q1);
    vector[p1] mu_magic1 = rep_vector(0, p1);
    matrix[p2, p2] magic2 = (Q2' * W2 * Q2);
    vector[p2] mu_magic2 = rep_vector(0, p2);

  // Compute, thin, and then scale QR decomposition
  matrix[N, K] X_Q = qr_Q(x)[, 1:K] * N;
  matrix[K, K] X_R = qr_R(x)[1:K, ] / N;
  matrix[K, K] X_R_inv = inverse(X_R);
}

parameters {
  real beta0;               // intercept
  vector[K] betas_tilde;    // coefficients
  vector[p1] phi;            // spatial effects
  vector[p2] phi22;            // spatial effects
  real<lower = 0> tau;      //scale for Ridge Regression
  real<lower = 0> r;
  real<lower = 0> r22;
  real<lower = 0> phi2_reciprocal;
}

transformed parameters {
  // Var(X) = mu + mu^2 / phi
  // phi -> infty no dispersion, phi -> 0 is full dispersion
  // this is 1 / phi, so that phi_reciprocal = 0 is no dispersion
  // https://statmodeling.stat.columbia.edu/2018/04/03/justify-my-love/
  real phi2 = 1. / sqrt(phi2_reciprocal + 0.001);
  vector[K] betas = X_R_inv * betas_tilde;

  real intercept = beta0;
  vector[N] fixedE = X_Q * betas_tilde;
  vector[N] randomE = Q1*phi + Q2*phi22;
  vector[N] mu_nb;

  mu_nb = intercept + fixedE + randomE;
}


model {
  beta0 ~ normal(0.0, 1.0);
  betas ~ normal(0.0, tau);
  tau ~ cauchy(0,1);

  phi2_reciprocal ~ normal(0.0, 1.0);

  phi ~ BSF_lpdf(inv(r), magic1, p1);
  phi22 ~ BSF_lpdf(inv(r22), magic2, p2);
  r ~ double_exponential(0,1);
  r22 ~ double_exponential(0,1);

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