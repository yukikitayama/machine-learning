# Boosting

## Extreme Gradient Boosting (XGBoost)

- Use `block structure` and parallel processing to reduce time complexity.
- Use `cache-aware prefetching algorithm` and `block size` make it possible to collect gradient statistics.
- Some data does not fit into main memory, it also stores the data on disk, but to make things fast, it compresses 
  blocks (`Block compression`) and shard the data onto multiple disks (`Block sharding`).
- `Out-of-core computation` and `cache-aware learning` was something new that XGBoost proposed. It allows the limited 
  computing resources to handle large scale data.
- `Weighted quantile sketch` was also new.
- Compared with `scikit-learn` and `R's gbm`, `XGBoost` runs faster and performs at the same accuracy as `scikit-learn`.
- XGBoost can work with terabyte size training data.
- In distributed setting, XGBoost runs faster than `Spark MLLib` and `H2O`.

## Reference

- Paper: XGBoost: A Scalable Tree Boosting System