# TagaSenti

Source: <https://huggingface.co/datasets/jjjardev/tagasenti>

- License: CC BY-SA 4.0
- File: `tagasenti_dataset.csv`
- SHA-256: `F9E189B33242B92235EE675AFFC7E4C06C7FD192972FE9E9FC0C24CAD6EB9CCF`
- Rows: 35,686
- Labels: 0 negative, 1 neutral, 2 positive

The dataset card describes a combination of human-labeled FiReCS and
SentiTaglish records, translated or synthetic examples, and LLM-labeled news and
social-media text. The downloadable CSV contains only `sentence` and `label`, so
individual rows cannot be separated reliably by their original provenance.

The local audit found 32,993 unique normalized texts, 2,693 internal duplicate
rows, 7,989 overlaps with FiReCS, and 24,988 texts beyond the combined
FiReCS/SentiTaglish normalized-text union. Matching FiReCS text includes 679
label conflicts.

TagaSenti is optional auxiliary training data, not human-gold evaluation data.
Training code gives FiReCS precedence, removes all normalized FiReCS test
overlaps, records source counts, and marks TagaSenti as mixed-provenance data in
the model metadata.
