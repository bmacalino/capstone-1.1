# SentiTaglish: Products and Services

Source: <https://huggingface.co/datasets/ccosme/SentiTaglishProductsAndServices>

- License: CC BY 4.0
- File: `SentiTaglish_ProductsAndServices.csv`
- SHA-256: `965040798AA172ABA316E9902303DA181EEAB9502DDE3DEC9B81BE4F6AFA2688`
- Rows: 10,510
- Labels: 1 negative, 2 neutral, 3 positive, 4 mixed

The copy supplied from `BRi002/SentiTaglishProductsAndServices` was verified as
byte-for-byte identical to the authoritative `ccosme` file above.

This corpus is an upgraded/re-annotated successor to FiReCS, not an independent
10,510-row addition. Normalized-text auditing found 10,459 overlaps with the
local FiReCS train/test files: 7,096 retain the same three-class label and 3,363
are labeled mixed. Only 45 normalized texts are absent from FiReCS.

The current DeFaketive model has three classes. Training code therefore excludes
mixed rows instead of silently mapping them to an incorrect class. FiReCS test
overlaps are always removed before fitting.
