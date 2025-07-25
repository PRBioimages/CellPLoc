# CellPLoc
Code for "CellPLoc: A Weakly-Supervised Multi-Instance Learning Framework for Protein Subcellular Localization and Heterogeneity Profiling in Single Cells"

Contact: Ying-Ying Xu, yyxu@smu.edu.cn

# Datasets
We trained our model on the HPA IF image database (release 21) and applied it to the HPA database (release 23).

# Model training
1. Prepare pre-segmented HPA single-cell images and save them (Name ID_idx.npy) to this address:/path/to/your/IF_single_cell_images/data/

2. Unzip and save /Meta_files/Meta_files.zip to /Meta_files

3. Run the code main_pseudo_softmatch.py
