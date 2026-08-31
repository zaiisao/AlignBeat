# Ablations

Every arm below is the **same command** with one argument changed. There is no
per-experiment script; if you want to reproduce a row, copy the base command and
substitute that row's flag.

## Base command

```sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=$PWD
python -u launch_scripts/train.py \
  --name NAME --head_type subset --gpu G --fold 0 --seed 0 \
  --max-epochs 20 --lr 3e-4 --batch-size 8 --accumulate-grad-batches 8 \
  --num-workers 6 --logger none --val-frequency 10000 --snapshot_every 3 \
  --n_min 172 --class_attention_layers 1 --omega_db 4 --learn_b \
  --init_encoder_from "$VANILLA" --freeze_encoder
```

`$VANILLA` = Analyze-SMC/third-party/beat_this/checkpoints/`vanilla_f0 S0 fold0 …ckpt`

`--freeze_encoder` is the **20-minute screen**: it holds the encoder fixed so only the
head trains. It tracks the end-to-end result closely (frozen 0.899 vs end-to-end
0.892) at a fraction of the cost. Use it before committing hours to anything.

## Results (held-out, split-half tuned; 300-song fold-0 val unless noted)

| arm | flag changed from base | joint | note |
|---|---|---|---|
| dense readout | `--head_type dense` | **0.919** | Beat This ceiling on this encoder |
| frozen_subset | *(base)* | 0.899 | best subset result |
| fsub_fix | *(base, post-1323a7e loss)* | running | ep2 0.836, ep5 0.856 — ahead of the 0.899 curve |
| ident_b | `--n_min 1500 --class_attention_layers 0 --b_scale 0.0045` (no `--learn_b`) | 0.825 | N = T, identity downsample |
| fpool_max | `--pool_mode max` | 0.813 | parameter-free pooling |
| fsub_tolflat | `--tol_flat 0.002355` | 0.816 | **negative result**, −0.083 |
| fpool_mean | `--pool_mode mean` | 0.802 | parameter-free pooling |

End-to-end (drop `--init_encoder_from`/`--freeze_encoder`, `--max-epochs 100`):

| arm | flag | joint |
|---|---|---|
| final100_dense | `--head_type dense` | 0.894 @ep99 |
| final100_subset | *(base)* | 0.892 @ep69 (unfinished) |
| r1_learnb | `--max-epochs 30` | 0.864 @ep29 |
| disc_lr2 | `--lr 8e-4 --head_lr 3e-4 --max-epochs 30` | 0.649 @ep17 (killed) |

## Scoring

```sh
POOL_MODE=<mean|max|""> ATTN_LAYERS=<n> NMIN=<n> LIMIT=300 \
  python /tmp/scoring/split_tune.py <gpu> "subset=<ckpt>"
```

Tunes decode (τ, δ) on one half of the val songs, reports on the disjoint half.
**Always use the same LIMIT** — mixing 150/200/300-song splits produces numbers that
are not comparable, which invalidated three comparisons during development.
