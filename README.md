# Bi-LSTM Splice Site Predictor

A comprehensive Machine Learning pipeline in Bioinformatics focused on identifying Exon and Intron boundaries (Splicing) directly from raw GenBank data.

## Pipeline Features
* **Smart Extraction:** Efficiently reads and parses massive `.gb` files using Biopython.
* **Advanced Biological Data Cleaning:** Automatically resolves reverse strand issues (Reverse Complement), filters out Single-Exon genes, and applies surgical sequence cropping to bypass the Promoter Region (5' UTR) trap.
* **Mutation Simulation:** Controlled injection of degenerate nucleotides for Data Augmentation and model robustness.
