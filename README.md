# Multi-Method Approach to membership selection in Ultra-Faint Dwarf Galaxies

This repository provides any file which were used during the analysis of RR Lyrae population in six Ultra-Faint Dwarf Galaxies: Boötes I, Boötes III, CarinaII, Coma Berenices, Sagittarius II, Ursa Major I.

Two complementary methodologies are applied to the $Gaia$ DR3 data: a classical approach and a statistical approach.


## How to Run the Project

1.  **Prerequisites**: Make sure you have Python installed, along with the necessary libraries. You can install them using pip:
    ```bash
    python3 -m pip install -r Clustering_pipeline/requirements.txt
    ```
2.  **Classical Approach** is based on proper motion analysis, Period-Wesenheit-Metallicity relation for RRLs, and color-magnitude diagram; it was performed with Jupyter Notebooks (in order of use: `BootesI.ipynb`, `BootesIII.ipynb`, `CarinaII.ipynb`, `ComaBerenices.ipynb`, `SagittariusII.ipynb`, `UrsaMajorI.ipynb`).
   - This part of the project is organized into the following directories:

     - **`Data/`**: You may have a look to `README_Data.md` file for what concerns the `Data/` directory in which should be present the initial datasets downloaded from $Gaia$ Archive (ADQL, https://gea.esac.esa.int/archive/).

     - **`Paper_tables/`**: This directory is where tables from literature papers of interest are stored.

     - **`Plot/`**: This directory is where the output plots are saved.

     - **`Output/`**: This directory is where the output files are saved.

3.  **Statistical Approach** used machine learning techniques (DBSCAN, HDBSCAN) to associate RRLs across the sky with their host galaxy based on astrometric and photometric properties; it was performed with a Python pipeline (see `Clustering_pipeline/` directory): 
   - Select sources with $G > 18 mag$ ('phot_g_mean_mag' > 18 mag) from `Data/vari_classifier_result.csv` catalogue by running `Clustering_18mag.ipynb`.
   - Run the main script in `Clustering_pipeline/` directory:
```bash
python main.py
```
   - This part of the project is organized into the following files and directories:
     - **`utils.py`**: This file contains all the utility functions needed for the project.

     - **`main.py`**: This is the main script that you run to execute the clustering algorithms.
    
     - **`model.py`**: This file contains all the Machine Learning model which can be performed, in particular we used 'apply_dbscan' and 'apply_hdbscan' functions.

     - **`output/`**: This directory is where the output files are saved.
  - Run `Clustering_UFDs.ipynb` to analyse the DBSCAN and HDBSCAN results computed by running the main script and to compare them with the RR Lyrae stars obtained as UFDs members with Classical Approach.
