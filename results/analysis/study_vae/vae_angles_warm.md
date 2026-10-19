# study_vae rung 1: plain VAE on the ten angles vs the OSDE density (parton level after trigger)

VAE: latent 10, hidden 128, warmup 20, free bits 0.1, 46 epochs, best validation −ELBO 12.170; f: 385 terms, y0 = 4.6e-06; events: {'tt_fit': 71322, 'tt_test': 7170, 'HH170': 73117, 'hh125': 10149}

## Event by event on the independent tt̄ sample (events with f > 0: 99.6%)

| quantity | mean |
|---|---|
| s_f | 11.893 |
| neg_elbo | 12.171 |
| iwae | 12.044 |
| kl | 3.241 |
| rec | 8.930 |
| s_flow | 12.256 |

| VAE score | Pearson r | Spearman | slope | intercept | residual mean ± std |
|---|---|---|---|---|---|
| neg_elbo | 0.721 | 0.671 | 0.739 | 3.38 | 0.278 ± 0.632 |
| iwae | 0.776 | 0.732 | 0.724 | 3.43 | 0.152 ± 0.543 |
| s_flow | 0.746 | 0.775 | 0.947 | 0.99 | 0.364 ± 0.709 |

Mean residual (−ELBO − s) in 8 quantile bins of each variable:

- Y: [0.4, 0.32, 0.21, 0.19, 0.13, 0.22, 0.29, 0.45]
- cos_Theta: [0.85, 0.48, 0.07, -0.16, -0.14, -0.01, 0.37, 0.77]
- cos_theta: [0.33, 0.25, 0.22, 0.24, 0.29, 0.3, 0.29, 0.3]
- phi: [0.26, 0.29, 0.3, 0.24, 0.26, 0.3, 0.31, 0.26]
- cos_theta_prime: [0.39, 0.31, 0.21, 0.18, 0.18, 0.22, 0.37, 0.35]
- phi_prime: [0.25, 0.26, 0.23, 0.36, 0.38, 0.23, 0.24, 0.28]
- cos_theta_tilde: [0.28, 0.28, 0.3, 0.25, 0.29, 0.25, 0.27, 0.3]
- phi_tilde: [0.29, 0.26, 0.25, 0.29, 0.27, 0.29, 0.3, 0.28]
- cos_theta_prime_tilde: [0.3, 0.32, 0.27, 0.22, 0.21, 0.23, 0.34, 0.32]
- phi_prime_tilde: [0.22, 0.24, 0.29, 0.26, 0.32, 0.35, 0.29, 0.26]

## AUC against the independent tt̄ sample

| score | HH170 | hh125 |
|---|---|---|
| s_f_shift | 0.670 ± 0.003 | 0.684 ± 0.004 |
| neg_elbo | 0.621 ± 0.003 | 0.638 ± 0.004 |
| iwae | 0.631 ± 0.003 | 0.648 ± 0.004 |
| kl | 0.545 ± 0.003 | 0.543 ± 0.004 |
| rec | 0.618 ± 0.003 | 0.640 ± 0.004 |
| s_flow | 0.680 ± 0.003 | 0.683 ± 0.004 |
