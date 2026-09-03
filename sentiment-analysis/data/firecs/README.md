# FiReCS evaluation data

This directory contains version 1.0 of the Filipino-English Reviews with
Code-Switching (FiReCS) dataset by Camilla Johnine Cosme and Marlene M. De
Leon. It is used to train and evaluate DeFaketive's three-class sentiment
labels.

- Source: https://huggingface.co/datasets/ccosme/FiReCS
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Citation: Cosme, C. J., & De Leon, M. M. (2024). *Sentiment Analysis of
  Code-Switched Filipino-English Product and Service Reviews Using
  Transformers-Based Large Language Models*. DOI: 10.1007/978-981-99-8349-0_11

The official train and test files are retained unchanged:

- `FiReCS_train_set.csv`: 7,340 rows; SHA-256
  `9CA5EF80B8F93836F5966A1DB8D017EBBD474E4B615468269CB0BA9D3778EB39`
- `FiReCS_test_set.csv`: 3,147 rows; SHA-256
  `6C172B8A52AD5D320D445BCF2229557DE103587673582767CC59F8FC42754875`

Label encoding is `0 = negative`, `1 = neutral`, and `2 = positive`. Use the
training set for model development. Keep the test set untouched until final
evaluation so benchmark results do not leak into model tuning.

The original CSV files above are never rewritten. The training pipeline
normalizes text in memory, removes two duplicate training rows, and excludes
four training texts also present in the official test split before fitting.
Model selection uses a stratified validation subset of the remaining training
data; test labels are not used for candidate selection.
