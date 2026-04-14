# Dataset Selection For The Thesis

## Thesis Goal

The thesis topic is:

- dynamic graph anomaly detection
- financial anti-fraud / anti-money-laundering
- generalizable model design, not a single-dataset hack

So the preferred datasets should satisfy as many of these conditions as possible:

1. financial transaction graph
2. explicit temporal information
3. public labels for illicit / licit / risk-related behavior
4. graph structure that can support dynamic GNN modeling

## Relevance Ranking

### Tier A: Directly matched to the thesis

- XinYe DGraph
  - already used in the project
  - financial anti-fraud task with temporal graph structure
- Elliptic++
  - Bitcoin transaction and wallet network
  - explicitly supports illicit transaction / illicit address detection
  - strong cross-domain value because it moves the thesis from domestic-style financial fraud to blockchain AML
- Elliptic Data Set
  - classic Bitcoin AML benchmark
  - same broad task family as Elliptic++
  - best candidate for the next supervised Bitcoin dataset

### Tier B: Very useful, but not the first add-on

- Elliptic2
  - much larger Bitcoin AML dataset
  - task is subgraph-level money-laundering detection rather than plain node classification
  - excellent for innovation, but requires a larger task/interface extension
- ORBITAAL
  - large temporal Bitcoin entity-transaction graph
  - best used for self-supervised temporal pretraining or representation learning
  - weaker as a direct supervised anti-fraud benchmark because the main value is graph scale rather than off-the-shelf fraud labels

### Tier C: Bitcoin-related but weaker for the thesis main line

- Bitcoin OTC / Bitcoin Alpha
  - temporal signed trust networks
  - useful for temporal graph robustness experiments
  - not ideal as the main anti-fraud benchmark because they are not direct illicit-transaction AML datasets

## Recommended Order

1. Keep XinYe as the main in-domain benchmark.
2. Add Elliptic++ as the first public Bitcoin AML dataset.
3. Add the original Elliptic Data Set as the second supervised Bitcoin benchmark.
4. If the thesis needs a stronger innovation point:
   - use ORBITAAL for self-supervised temporal pretraining, or
   - extend the pipeline to Elliptic2 for subgraph-level AML detection.

## What This Means For The Codebase

- generic node-level pipeline:
  - XinYe
  - Elliptic++
  - Elliptic Data Set
- extended research branch:
  - Elliptic2
  - ORBITAAL pretraining

This keeps the core engineering architecture stable while leaving room for a stronger final thesis story.

## Source Links

- Elliptic++ official repo:
  - https://github.com/git-disl/EllipticPlusPlus
- Elliptic Data Set official announcement:
  - https://www.elliptic.co/media-center/elliptic-releases-bitcoin-transactions-data
- Elliptic2 official guide:
  - https://github.com/MITIBMxGraph/Elliptic2
- ORBITAAL paper:
  - https://arxiv.org/abs/2408.14147
- Bitcoin OTC official SNAP page:
  - https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
