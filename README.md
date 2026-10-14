# sgd_gmm

Reference implementation of

> Pasande, Hosseini, Araabi.
> *Stochastic first-order learning for large-scale flexibly-tied Gaussian mixture models.*
> Pattern Recognition Letters, 2024.
> https://doi.org/10.1016/j.patrec.2023.12.021

## Model

Flexibly-tied covariances share one factor `U`:

```text
precision_k = lambda_k * U * diag(d_k) * U^T
```

- `d_k` is SoftPlus of unconstrained parameters (`omega = 1`).
- Mixture weights are a SoftMax over logits `alpha`.
- Training loss is negative log-likelihood; Wishart / Gamma / Gaussian-mean /
  Dirichlet regularizers are available but **off by default**.

| `method=` | Notes |
|---|---|
| `adam` | free `U` |
| `acclip` | free `U`; Euclidean adaptive coordinate clipping |
| `riemannian_acclip` (proposed) | Riemannian ACClip on `SO(n)` × Euclidean |

With `orthogonal=True`, `U` stays on the orthogonal group via tangent-space
projection and a retraction (`qr` default, or `polar`/`svd`, or `cayley`).
With `orthogonal=False`, `U` is free and the log-density includes `log|det U|`.

## Layout

```text
src/sgd_gmm/
  model.py
  regularizers.py
  manifold.py
  optimizers/acclip.py
  optimizers/riemannian.py
  train.py
```

## Usage

```python
from sgd_gmm import FlexiblyTiedGMM, train

device = "cpu"  # or "cuda"
model = FlexiblyTiedGMM(dimension=21, n_components=21, orthogonal=True, device=device)
model.initialize(train_data)  # train_data: (N, D) tensor on `device`
result = train(
    model,
    train_data,
    method="riemannian_acclip",
    epochs=100,
    batch_size=128,
    learning_rate=1e-2,
)
```

Retraction / projection for Riemannian ACClip:

```python
from sgd_gmm.optimizers import RiemannianACClip
from sgd_gmm.optimizers.riemannian import split_gmm_param_groups

opt = RiemannianACClip(
    split_gmm_param_groups(
        model,
        lr=0.07,
        retraction="qr",    # or "polar" / "svd" / "cayley"
        projection="half",  # or "simple"
        mod=1,              # 0 = ACCLip only
    )
)
```

