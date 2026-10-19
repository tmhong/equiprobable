# study_vae rung 1: plain VAE on the ten angles vs the OSDE density (parton level after trigger)

VAE: latent 10, hidden 128, 30 epochs, best validation −ELBO 12.103; f: 385 terms, y0 = 4.6e-06; events: {'tt_fit': 71322, 'tt_test': 7170, 'HH170': 73117, 'hh125': 10149}

## Event by event on the independent tt̄ sample (events with f > 0: 99.6%)

| quantity | mean |
|---|---|
| s_f | 11.893 |
| neg_elbo | 12.083 |
| iwae | 12.081 |
| kl | 0.039 |
| rec | 12.044 |

| VAE score | Pearson r | Spearman | slope | intercept | residual mean ± std |
|---|---|---|---|---|---|
| neg_elbo | 0.761 | 0.718 | 0.712 | 3.61 | 0.190 ± 0.562 |
| iwae | 0.763 | 0.720 | 0.712 | 3.61 | 0.188 ± 0.559 |

Mean residual (−ELBO − s) in 8 quantile bins of each variable:

- Y: [0.39, 0.24, 0.08, 0.07, 0.03, 0.11, 0.23, 0.38]
- cos_Theta: [0.71, 0.35, -0.06, -0.27, -0.23, -0.06, 0.34, 0.74]
- cos_theta: [0.18, 0.16, 0.16, 0.2, 0.22, 0.21, 0.2, 0.18]
- phi: [0.19, 0.19, 0.19, 0.18, 0.17, 0.22, 0.22, 0.16]
- cos_theta_prime: [0.42, 0.25, 0.13, 0.1, 0.07, 0.1, 0.24, 0.2]
- phi_prime: [0.1, 0.19, 0.18, 0.25, 0.25, 0.21, 0.2, 0.15]
- cos_theta_tilde: [0.19, 0.19, 0.2, 0.21, 0.2, 0.16, 0.18, 0.2]
- phi_tilde: [0.21, 0.17, 0.15, 0.17, 0.18, 0.2, 0.22, 0.22]
- cos_theta_prime_tilde: [0.25, 0.2, 0.12, 0.11, 0.14, 0.18, 0.29, 0.23]
- phi_prime_tilde: [0.14, 0.17, 0.19, 0.18, 0.23, 0.23, 0.21, 0.15]

## AUC against the independent tt̄ sample

| score | HH170 | hh125 |
|---|---|---|
| s_f_shift | 0.670 ± 0.003 | 0.684 ± 0.004 |
| neg_elbo | 0.619 ± 0.003 | 0.636 ± 0.004 |
| iwae | 0.620 ± 0.003 | 0.636 ± 0.004 |
| kl | 0.599 ± 0.003 | 0.606 ± 0.004 |
| rec | 0.619 ± 0.003 | 0.636 ± 0.004 |
